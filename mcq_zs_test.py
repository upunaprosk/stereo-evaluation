import os
import re
import sys
import json
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # BitsAndBytesConfig is optional (load_in_8bit)
import argparse
from datasets import load_dataset
import datetime

def seed_all(seed):
    import random
    import os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

model_name = ""
apply_chat_template = False
prompt_prefix = 'Answer the following multiple-choice question with the answer letter. Then after newline provide optional explanation.'
print_generated_tokens = 0
uppercase_unknown = False

def optional_apply_chat_template(prompt, tokenizer):
    if apply_chat_template:
        # Note: for Qwen3-0.6B chat template improves UNQOVER a bit, for Llama-3.1-8B-Instruct it's harmful
        messages = [{"role": "user", "content": prompt}]
        if prompt_prefix != "":
            messages.append({"role": "system", "content": prompt_prefix})
        # messages.append({"role": "user", "content": prompt}) # for Qwen3-0.6B it's better to put system prompt in the end
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            date_string=datetime.date.today().strftime("%d %b %Y") # fix "Today Date: 26 Jul 2024" for Llama tokenizer
        )
    return prompt

def add_prompt_prefix(prompt):
    if prompt_prefix != "" and not apply_chat_template:
        prompt = f"{prompt_prefix}\n\n{prompt}"
    return prompt

def get_prediction(logits, label, prompt, tokenizer, n=8, verbose=False):
    """
    Find the most probable token in logits. Consider several token options for each answer letter.
    """
    answers = "ABCDEFGH"[:n]
    answer_ids = []
    for ans in answers:
        # Try three tokens for each answer ('A' and ' A' are different tokens)
        # (for Llama 3.1 8B Instruct all three token ids are different)
        answer_ids.append(tokenizer(f"Answer: {ans}").input_ids[-1])
        answer_ids.append(tokenizer(prompt + ans).input_ids[-1])
        answer_ids.append(tokenizer(ans).input_ids[-1])

    all_probs = torch.nn.functional.softmax(logits.float(), dim=0).detach().cpu().numpy()
    answer_probs = (
        torch.nn.functional.softmax(
            torch.tensor([float(all_probs[x]) for x in answer_ids]).float(),
            dim=0
        ).detach().cpu().numpy()
    )
    pred = np.argmax(answer_probs) // 3 # 3 possible token ids for each answer
    cor = pred == label

    if verbose:
        print_logits = [float(logits[x]) for x in answer_ids]
        print("MODEL_OUTPUT:", answers[pred],
              cor, 'ANSWER:',  answers[label],
              f" {label=} {print_logits=}", file=sys.stderr)

    return pred

def generate_text(model, tokenizer, inputs, max_len):
    # top_k, top_p, temperature -- use model's default (generation_config.js)
    return model.generate(input_ids=inputs.input_ids.to(model.device),
                          attention_mask=inputs.attention_mask.to(model.device),
                          do_sample=True,
                          max_length=max_len,
                          temperature=0.0001,
                          num_return_sequences=1,
                          pad_token_id=tokenizer.eos_token_id) # [:, inputs.input_ids.shape[-1]:]

def format_arc(dataset, idx):
    choices = ['A', 'B', 'C', 'D', 'E']
    question = dataset[idx]['question']
    answers = dataset[idx]['choices']['text']
    prompt = f'{question}'
    prompt = add_prompt_prefix(prompt)
    for i in range(len(answers)):
        prompt += f'\n{choices[i]}. {answers[i]}'
    prompt += '\nAnswer:'

    return prompt

def arc_evaluate(tag, model, tokenizer, category, persistent_dir='bias_bench-main', verbose=False):
    print(f'--------------Evaluate ARC {category}--------------')
    if category == 'easy':
        # dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', cache_dir='/data/yichenli/.cache/huggingface/datasets')['test']
        dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy')['test']
    else:
        # dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge', cache_dir='/data/yichenli/.cache/huggingface/datasets')[
        dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge')['test']

    cors = []

    for i in tqdm(range(len(dataset)), desc="arc", file=sys.stdout):
        prompt = format_arc(dataset, i)
        prompt = optional_apply_chat_template(prompt, tokenizer)
        if verbose:
            print(f"PROMPT: {prompt}", file=sys.stderr)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}[dataset[i]['answerKey']]
        if print_generated_tokens > 0:
            max_len = input_ids.shape[-1] + print_generated_tokens

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1]
            if print_generated_tokens > 0:
                model_gen_tokens = generate_text(model, tokenizer, inputs, max_len)

        if print_generated_tokens > 0:
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True).strip()
            print(f"Model answer: <{model_gen_str}>", file=sys.stderr)

        pred = get_prediction(logits, label, prompt, tokenizer, verbose=verbose)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)

    print("Average accuracy {:.5f} - {}".format(acc, category))

    result_dict = {}

    result_dict['total_acc'] = acc
    os.makedirs(f"{persistent_dir}/results/test/arc_{category}/layer", exist_ok=True)
    with open(f"{persistent_dir}/results/test/arc_{category}/layer/{tag}.json", "w") as f:
        json.dump(result_dict, f)

def unqover_evaluate(tag, model, tokenizer, size=None, persistent_dir='bias_bench-main', verbose=False):
    if verbose:
        print(f"----------Evaluate UNQOVER-------------", file=sys.stderr)
    # dataset = load_dataset('json', data_files=f'{os.getenv("HOME")}/github/CAL/data/unqover/datas.jsonl')['train']
    url = "https://raw.githubusercontent.com/spirit-moon-fly/CAL/master/data/unqover/datas.jsonl"
    dataset = load_dataset("json", data_files={"train": url})["train"]
    # frame = load_dataset('json', data_files=f'CAL_data/unqover/datas.jsonl')['train']
    # print("Using UnQover with the corrected Unknown capitalization")
    # frame = load_dataset("iproskurina/unqover")["train"]
    #
    def format_unqover(dataset, idx):
        choices = "ABC"
        context = dataset[idx]['context']
        question = dataset[idx]['question']
        answers = [dataset[idx]['ans0'], dataset[idx]['ans1'], dataset[idx]['ans2']]
        prompt = f'{context} {question}'
        prompt = add_prompt_prefix(prompt)
        for i in range(len(answers)):
            if uppercase_unknown and answers[i] == "unknown":
                answers[i] = "Unknown"
            prompt += f'\n{choices[i]}. {answers[i]}'
        prompt += '\nAnswer:'
        return prompt

    cors = []

    if size is None:
        size = len(dataset)
    else:
        size = min(size, len(dataset))
    for i in tqdm(range(size), desc="unqover", file=sys.stdout):
        prompt = format_unqover(dataset, i)
        prompt = optional_apply_chat_template(prompt, tokenizer)
        label = dataset[i]['label']
        if verbose:
            print(f"PROMPT: {prompt}", file=sys.stderr)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        if print_generated_tokens > 0:
            max_len = input_ids.shape[-1] + print_generated_tokens

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1]
            if print_generated_tokens > 0:
                model_gen_tokens = generate_text(model, tokenizer, inputs, max_len)

        if print_generated_tokens > 0:
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True).strip()
            print(f"Model answer: <{model_gen_str}>", file=sys.stderr)

        pred = get_prediction(logits, label, prompt, tokenizer, n=3, verbose=verbose)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)

    print("Average accuracy {:.5f}".format(acc))

    result_dict = {}

    result_dict['total_acc'] = acc
    os.makedirs(f"{persistent_dir}/results/test/unqover/layer", exist_ok=True)
    with open(f"{persistent_dir}/results/test/unqover/layer/{tag}.json", "w") as f:
        json.dump(result_dict, f)

BBQ_CATEGORY = {
    'age': 'Age',
    'disability': 'Disability_status',
    'gender': 'Gender_identity',
    'nationality': 'Nationality',
    'physical-appearance': 'Physical_appearance',
    'ethnicity': 'Race_ethnicity',
    'race-x-ses': 'Race_x_SES',
    'race-x-gender': 'Race_x_gender',
    'religion': 'Religion',
    'socioeconomic': 'SES',
    'sexual-orientation': 'Sexual_orientation',
    'all': 'All'
}

def bbq_evaluate(tag, model, tokenizer, bias_type, size=None, persistent_dir='bias_bench-main', verbose=False):
    category = BBQ_CATEGORY[bias_type]
    if verbose:
        print(f"----------Preprocess BBQ: {category}-------------", file=sys.stderr)
    frame = load_dataset("oskarvanderwal/bbq", category)['test']

    frame = list(frame)
    random.shuffle(frame)

    def format_bbq(dataset, idx):
        choices = ['A', 'B', 'C', 'D', 'E']
        context = dataset[idx]['context']
        question = dataset[idx]['question']
        answers = [dataset[idx]['ans0'], dataset[idx]['ans1'], dataset[idx]['ans2']]
        prompt = f'{context} {question}'
        prompt = add_prompt_prefix(prompt)
        for i in range(len(answers)):
            prompt += f'\n{choices[i]}. {answers[i]}'
        prompt += '\nAnswer:'
        return prompt

    cors = []

    if size is None:
        size = len(frame)
    else:
        size = min(size, len(frame))
    for i in tqdm(range(size), desc="bbq", file=sys.stdout):
        label = frame[i]['label']
        prompt = format_bbq(frame, i)
        prompt = optional_apply_chat_template(prompt, tokenizer)
        if verbose:
            print(f"PROMPT: {prompt}", file=sys.stderr)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        if print_generated_tokens > 0:
            max_len = input_ids.shape[-1] + print_generated_tokens

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1]
            if print_generated_tokens > 0:
                model_gen_tokens = generate_text(model, tokenizer, inputs, max_len)

        if print_generated_tokens > 0:
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True).strip()
            print(f"Model answer: <{model_gen_str}>", file=sys.stderr)

        pred = get_prediction(logits, label, prompt, tokenizer, n=3, verbose=verbose)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)

    print("Average accuracy {:.5f} - {}".format(acc, bias_type))

    result_dict = {}

    result_dict['total_acc'] = acc
    os.makedirs(f"{persistent_dir}/results/test/bbq_{bias_type}/layer", exist_ok=True)
    with open(f"{persistent_dir}/results/test/bbq_{bias_type}/layer/{tag}.json", "w") as f:
        json.dump(result_dict, f)

def bbq_helm_evaluate(tag, model, tokenizer, bias_type, persistent_dir='bias_bench-main', verbose=False):
    category = BBQ_CATEGORY[bias_type]
    if category == "All":
        category = "all"
    if verbose:
        print(f"----------Preprocess BBQ_HELM: {category}-------------", file=sys.stderr)
    frame = load_dataset("lighteval/bbq_helm", category)['test']

    frame = list(frame)
    random.shuffle(frame)

    def format_bbq_helm(dataset, idx):
        choices = ['A', 'B', 'C', 'D', 'E']
        context = dataset[idx]['context']
        question = dataset[idx]['question']
        answers = dataset[idx]['choices']
        prompt = f'{context} {question}'
        prompt = add_prompt_prefix(prompt)
        for i in range(len(answers)):
            prompt += f'\n{choices[i]}. {answers[i]}'
        prompt += '\nAnswer:'
        return prompt

    cors = []

    for i in tqdm(range(len(frame)), desc="bbq_helm", file=sys.stdout):
        label = frame[i]['gold_index']
        prompt = format_bbq_helm(frame, i)
        prompt = optional_apply_chat_template(prompt, tokenizer)
        if verbose:
            print(f"PROMPT: {prompt}", file=sys.stderr)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        if print_generated_tokens > 0:
            max_len = input_ids.shape[-1] + print_generated_tokens

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1]
            if print_generated_tokens > 0:
                model_gen_tokens = generate_text(model, tokenizer, inputs, max_len)

        if print_generated_tokens > 0:
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True).strip()
            print(f"Model answer: <{model_gen_str}>", file=sys.stderr)

        pred = get_prediction(logits, label, prompt, tokenizer, n=3, verbose=verbose)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)

    print("Average accuracy {:.5f} - {}".format(acc, bias_type))

    result_dict = {}

    result_dict['total_acc'] = acc
    os.makedirs(f"{persistent_dir}/results/test/bbq_{bias_type}/layer", exist_ok=True)
    with open(f"{persistent_dir}/results/test/bbq_{bias_type}/layer/{tag}.json", "w") as f:
        json.dump(result_dict, f)


def format_mmlu(dataset, idx, include_answer=True):
    choices = ['A', 'B', 'C', 'D']
    question = dataset[idx]['question']
    answers = dataset[idx]['choices']
    label = dataset[idx]['answer']
    prompt = f'{question}'
    for i in range(len(answers)):
        prompt += f'\n{choices[i]}. {answers[i]}'
    prompt += '\nAnswer:'
    if include_answer:
        prompt += f' {choices[label]}\n\n'
    return prompt

def format_mmlu_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def gen_mmlu_prompt(dataset, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about{format_mmlu_subject(subject)}.\n\n"
    if k == -1:
        k = len(dataset)
    for i in range(k):
        prompt += format_mmlu(dataset, i, include_answer=True)
    return prompt


def mmlu_evaluate_a_subject(model, tokenizer, subject, size=None, verbose=False):
    dataset = load_dataset("cais/mmlu", subject)['test']
    dev_dataset = load_dataset("cais/mmlu", subject)['dev']

    def id(head_output, layer_name):
        return head_output

    cors = []

    if size is None:
        size = len(dataset)
    else:
        size = min(size, len(dataset))
    for i in tqdm(range(size), desc=f"mmlu_{subject}", file=sys.stdout):
        k = 5
        prompt_end = format_mmlu(dataset, i, include_answer=False)
        train_prompt = gen_mmlu_prompt(dev_dataset, subject, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer(prompt, return_tensors="pt")

        while inputs.input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_mmlu_prompt(dataset, subject, k)
            prompt = train_prompt + prompt_end
            inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        label = dataset[i]['answer']

        if print_generated_tokens > 0:
            max_len = input_ids.shape[-1] + print_generated_tokens

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1]
            if print_generated_tokens > 0:
                model_gen_tokens = generate_text(model, tokenizer, inputs, max_len)

        if print_generated_tokens > 0:
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True).strip()
            print(f"Model answer: <{model_gen_str}>", file=sys.stderr)

        pred = get_prediction(logits, label, prompt, tokenizer, n=4, verbose=verbose)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def mmlu_evaluate(tag, model, tokenizer, category="all", size=None, persistent_dir="bias_bench-main", verbose=False):
    print(f'--------------Evaluate MMLU {category}--------------')
    if category == 'all':
        subjects = subcategories.keys()
        all_cors = []
        subcat_cors = {
            subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
        }
        cat_cors = {cat: [] for cat in categories}
        for subject in subjects:
            cors = mmlu_evaluate_a_subject(model, tokenizer, subject, size=size, verbose=verbose)
            subcats = subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(cors)
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key].append(cors)
            all_cors.append(cors)

        results = {"subcategories": {}, "categories": {}}
        for subcat in subcat_cors:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            results["subcategories"][subcat] = subcat_acc
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

        for cat in cat_cors:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            results["categories"][cat] = cat_acc
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        weighted_acc = np.mean(np.concatenate(all_cors))
        results["weighted_accuracy"] = weighted_acc
        print("Average accuracy: {:.3f}".format(weighted_acc))

        os.makedirs(f"{persistent_dir}/results/test/mmlu", exist_ok=True)
        with open(f"{persistent_dir}/results/test/mmlu/{tag}.json", "w") as f:
            json.dump(results, f)

    elif category in subcategories.keys():
        cors = mmlu_evaluate_a_subject(model, tokenizer, category, size=size, verbose=verbose)


def main(args):
    global model_name
    model_name = args.model_path.split("/")[-1]

    use_bnb_8bit = False
    # if not "-0.6B" in model_name and not "-0.5B" in model_name and not "-1B-" in model_name and model_name != "gemma-2b-it" and not "GPTQ" in model_name and not "gptq" in model_name and not "-int4-" in model_name and not 'Sparse-' in model_name:
    #     use_bnb_8bit = True

    options = {
        "torch_dtype": torch.float16,
    }

    if use_bnb_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        options["quantization_config"] = quantization_config
        print(f"Loading in 8 bit: BitsAndBytesConfig(load_in_8bit=True)", file=sys.stderr)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True, 
        device_map="cuda",
        **options
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    persistent_dir = "bias_bench-main"

    for test in args.tests.split(","):
        if test.startswith("arc_"):
            arc_evaluate(model_name,
                         model,
                         tokenizer,
                         test.replace("arc_", ""),
                         persistent_dir=persistent_dir,
                         verbose=True)

        if test.startswith("unqover"):
            size = None
            if test.startswith("unqover_"):
                size = int(test.split("_")[-1])
            unqover_evaluate(model_name,
                             model,
                             tokenizer,
                             size=size,
                             persistent_dir=persistent_dir,
                             verbose=True)

        if test.startswith("bbq_helm"):
            bias_type = "all"
            if test.startswith("bbq_helm_"):
                bias_type = test.replace("bbq_helm_", ""),
            bbq_helm_evaluate(model_name,
                         model,
                         tokenizer,
                         bias_type,
                         persistent_dir=persistent_dir,
                         verbose=True)

        if test.startswith("bbq") and not test.startswith("bbq_helm"):
            size = None
            if re.search(r"_\d+$", test):
                size = int(test.split("_")[-1])
                test = "_".join(test.split("_")[:-1])
            bias_type = "all"
            if test.startswith("bbq_"):
                bias_type = test.replace("bbq_", ""),
            bbq_evaluate(model_name,
                         model,
                         tokenizer,
                         bias_type,
                         size=size,
                         persistent_dir=persistent_dir,
                         verbose=True)
        if test.startswith("mmlu"):
            size = None
            if re.search(r"_\d+$", test):
                size = int(test.split("_")[-1])
                test = "_".join(test.split("_")[:-1])
            category = "all"
            if test.startswith("mmlu_"):
                category = test.replace("mmlu_", "")
            mmlu_evaluate(
                model_name,
                model,
                tokenizer,
                category=category,
                size=size,
                persistent_dir=persistent_dir,
                verbose=True)


if __name__ == '__main__':

    torch.inference_mode()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_path', type=str,
        help='Huggingface model to load (or local folder)'
    )
    parser.add_argument(
        '--tests', type=str,
        help='Tests to run (arc_easy, arc_challenge, bbq, bbq_*, bbq_helm, bbq_helm_*, unqover, mmlu, mmlu_*)'
    )
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--apply_chat_template', action="store_true")
    parser.add_argument('--prompt_prefix', type=str, default="default")
    parser.add_argument('--print_generated_tokens', type=int, default=0)
    parser.add_argument('--uppercase_unknown', action="store_true")

    args = parser.parse_args()
    seed_all(args.seed)
    apply_chat_template = args.apply_chat_template
    if args.prompt_prefix != "default":
        prompt_prefix = args.prompt_prefix
    print_generated_tokens = args.print_generated_tokens
    uppercase_unknown = args.uppercase_unknown
    print(f"Prompting options: {apply_chat_template=} {uppercase_unknown=} {print_generated_tokens=} {prompt_prefix=}", file=sys.stderr)

    main(args)
