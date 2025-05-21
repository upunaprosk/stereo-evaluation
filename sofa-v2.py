import os
import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
import json
import numpy as np
from tqdm import tqdm

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

def generate_identity_terms_json():
    os.makedirs("data/sofa", exist_ok=True)
    gender_df = pd.read_csv("identity_terms/gender.csv")
    sexuality_df = pd.read_csv("identity_terms/sexuality.csv")
    race_df = pd.read_csv("identity_terms/race.csv")
    countries_df = pd.read_csv("identity_terms/countries.csv")
    religion_df = pd.read_csv("identity_terms/religion.csv")
    disability_df = pd.read_csv("identity_terms/disability.csv")
    identity_terms = {
        'gender': gender_df['TERM'].dropna().tolist() + sexuality_df['TERM'].dropna().tolist(),
        'race': race_df['TERM'].dropna().tolist() + countries_df['COUNTRY_ADJ'].dropna().tolist(),
        'culture': religion_df['TERM'].dropna().tolist(),
        'disabled': disability_df['TERM'].dropna().tolist()
    }
    with open("data/sofa/identity_terms.json", 'w') as f:
        json.dump(identity_terms, f, indent=2)
    print("identity_terms.json has been created successfully!")

def download_and_save_dataset():
    print("SBIC-Pro.csv not found. Downloading dataset...")
    ds = load_dataset("copenlu/sofa", "default")
    df = pd.DataFrame(ds['train'])
    os.makedirs("data/sofa", exist_ok=True)
    df.to_csv("data/sofa/SBIC-Pro.csv", index=False)
    print("SBIC-Pro.csv has been downloaded and saved!")

def compute_perplexity(predictions, model_name, batch_size=16, device=None, max_length=512):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(predictions, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppls = []
    for start_index in tqdm(range(0, len(encoded_texts), batch_size), desc=f"Processing {model_name}"):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]
        labels = encoded_batch
        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls.extend(perplexity_batch.tolist())
    return ppls

def compute_probe_ppl(df, model_name, batch_size=32):
    df['probe'] = df['probe'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    input_texts = df['probe'].tolist()
    batch_perplexities_dict = {model_name: []}
    if "llama" in model_name or "gpt" in model_name or "meta" in model_name:
        for i in tqdm(range(0, len(input_texts), batch_size), desc=f"Processing {model_name} probes"):
            input_text_batch = input_texts[i:i + batch_size]
            batch_perplexities = compute_perplexity(input_text_batch, model_name)
            batch_perplexities_dict[model_name].extend(batch_perplexities)
    else:
        perplexity = load("perplexity", module_type="metric")
        for i in tqdm(range(0, len(input_texts), batch_size), desc=f"Processing {model_name} probes"):
            input_text_batch = input_texts[i:i + batch_size]
            batch_perplexities = perplexity.compute(model_id=model_name, predictions=input_text_batch)['perplexities']
            batch_perplexities_dict[model_name].extend(batch_perplexities)
    np.save(f'perplexities_{model_name.replace("/", "-")}.npy', np.array(batch_perplexities_dict[model_name]))
    return batch_perplexities_dict

def compute_identity_ppl(identity_terms, model_name, batch_size=32):
    PPL_identity = {model_name: []}
    for identity, terms in identity_terms.items():
        batch_perplexities = compute_perplexity(terms, model_name, batch_size=batch_size)
        PPL_identity[model_name].extend(batch_perplexities)
    return PPL_identity

def compute_ratio_and_log(probe_ppl, identity_ppl, model_name):
    ratio_results = {}
    ratio_results[model_name] = [probe / identity if identity != 0 else 0 for probe, identity in zip(probe_ppl[model_name], identity_ppl[model_name])]
    ratio_results[model_name] = np.log10(ratio_results[model_name])
    return ratio_results

def analyze_and_process(input_file, identity_terms_file, output_file, model_name, batch_size=32):
    df = pd.read_csv(input_file)
    with open(identity_terms_file, 'r') as f:
        identity_terms = json.load(f)
    probe_ppl = compute_probe_ppl(df, model_name, batch_size)
    identity_ppl = compute_identity_ppl(identity_terms, model_name, batch_size)
    ratio_results = compute_ratio_and_log(probe_ppl, identity_ppl, model_name)
    result_df = pd.DataFrame(ratio_results)
    result_df['id'] = df['id']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return result_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/sofa/SBIC-Pro.csv")
    parser.add_argument("--identity_terms_file", type=str, default="data/sofa/identity_terms.json")
    parser.add_argument("--output_file", type=str, default="data/sofa/SBIC-Pro-w-Ratio-PPLs.csv")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        download_and_save_dataset()
    if not os.path.exists(args.identity_terms_file):
        print("identity_terms.json not found. Generating it...")
        generate_identity_terms_json()
    analyze_and_process(
        input_file=args.input_file,
        identity_terms_file=args.identity_terms_file,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()