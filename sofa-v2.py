import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn import CrossEntropyLoss
from evaluate import load as hf_load
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

def compute(predictions, model_id, batch_size=16, add_start_token=True, device=None, max_length=None):
    if device is not None:
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if add_start_token and max_length:
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if add_start_token:
        assert torch.all(torch.ge(attention_mask.sum(1), 1))
    else:
        assert torch.all(torch.ge(attention_mask.sum(1), 2))

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start in tqdm(range(0, len(input_ids), batch_size), desc=f"Computing PPL - {model_id}"):
        end = min(start + batch_size, len(input_ids))
        input_batch = input_ids[start:end]
        mask_batch = attention_mask[start:end]

        if add_start_token:
            bos = torch.tensor([[tokenizer.bos_token_id]] * input_batch.size(0)).to(device)
            input_batch = torch.cat([bos, input_batch], dim=1)
            mask_batch = torch.cat([torch.ones_like(bos), mask_batch], dim=1)

        labels = input_batch

        with torch.no_grad():
            logits = model(input_batch, attention_mask=mask_batch).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask_batch[..., 1:].contiguous()

        loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_mask).sum(1) / shift_mask.sum(1)
        ppls += torch.exp(loss).tolist()

    return ppls

def compute_ppl_for_model(predictions, model_name, batch_size=64):
    use_custom = any(k in model_name.lower() for k in ["llama", "meta", "bloom"])
    if use_custom:
        return compute(predictions=predictions, model_id=model_name, batch_size=batch_size)
    else:
        metric = hf_load("perplexity", module_type="metric")
        try:
            result = metric.compute(model_id=model_name, predictions=predictions, batch_size=batch_size)
            ppls = result.get("perplexities", [])
            if not ppls or all(np.isnan(p) for p in ppls):
                print(f"[{model_name}] WARNING: metric.compute() returned empty or NaN-only results")
                return [float('nan')] * len(predictions)
            return ppls
        except Exception as e:
            print(f"[{model_name}] ERROR in metric.compute(): {e}")
            return [float('nan')] * len(predictions)


def compute_probe_ppl(df, model_name, batch_size=64):
    df['probe'] = df['probe'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    texts = df['probe'].dropna().tolist()
    all_ppls = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            ppls = compute_ppl_for_model(batch, model_name, batch_size=batch_size)
        except Exception as e:
            print(f"[!] Batch error at {i}: {e}")
            ppls = [float("nan")] * len(batch)
        all_ppls.extend(ppls)
    return {model_name: all_ppls}

def compute_identity_ppl(identity_terms_dict, model_name, batch_size=64):
    group_ppl = {}
    all_ppls = []
    for group, terms in identity_terms_dict.items():
        filtered = [t.capitalize() for t in terms if isinstance(t, str) and len(t.split()) > 1]
        if not filtered:
            group_ppl[group] = []
            continue
        try:
            ppls = compute_ppl_for_model(filtered, model_name, batch_size=batch_size)
        except Exception as e:
            print(f"[!] Group error ({group}): {e}")
            ppls = [float("nan")] * len(filtered)
        group_ppl[group] = ppls
        all_ppls.extend(ppls)
        print(f"{group}: {np.nanmean(ppls):.2f}")
    return {model_name: all_ppls}, group_ppl

def compute_ratio_and_log(probe_ppl, group_ppl, model_name):
    log_ratios = {}
    probe_vals = np.array(probe_ppl[model_name])
    for group, identities in group_ppl.items():
        identity_vals = np.array(identities)
        min_len = min(len(probe_vals), len(identity_vals))
        valid = (identity_vals[:min_len] > 0) & (probe_vals[:min_len] > 0)
        if not np.any(valid):
            log_ratios[group] = np.array([])
            continue
        ratio = probe_vals[:min_len][valid] / identity_vals[:min_len][valid]
        log_ratios[group] = np.log10(ratio)
        print(f"[{model_name}] {group}: avg log10(P/I) = {np.mean(log_ratios[group]):.4f}")
    return log_ratios

def analyze(model_name, probe_file, identity_file, output_file, batch_size=64):
    df = pd.read_csv(probe_file)
    with open(identity_file, 'r') as f:
        identity_terms = json.load(f)
    probe_ppl = compute_probe_ppl(df, model_name, batch_size)
    identity_ppl, group_ppl = compute_identity_ppl(identity_terms, model_name, batch_size)
    ratio = compute_ratio_and_log(probe_ppl, group_ppl, model_name)
    df_ratio = pd.DataFrame(ratio)
    df_ratio["id"] = df["id"][:len(df_ratio)]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_ratio.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def generate_identity_terms_json(identity_dir, output_path):
    gender_df = pd.read_csv(os.path.join(identity_dir, "gender.csv"))
    sexuality_df = pd.read_csv(os.path.join(identity_dir, "sexuality.csv"))
    race_df = pd.read_csv(os.path.join(identity_dir, "race.csv"))
    countries_df = pd.read_csv(os.path.join(identity_dir, "countries.csv"))
    religion_df = pd.read_csv(os.path.join(identity_dir, "religion.csv"))
    disability_df = pd.read_csv(os.path.join(identity_dir, "disability.csv"))

    identity_terms = {
        "gender": gender_df["TERM"].dropna().tolist() + sexuality_df["TERM"].dropna().tolist(),
        "race": race_df["TERM"].dropna().tolist() + countries_df["COUNTRY_ADJ"].dropna().tolist(),
        "culture": religion_df["TERM"].dropna().tolist(),
        "disabled": disability_df["TERM"].dropna().tolist()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(identity_terms, f, indent=2)
    print(f"Generated identity file at {output_path}")

def download_sofa_dataset(probe_file):
    print(f"{probe_file} not found. Downloading SOFA dataset...")
    ds = load_dataset("copenlu/sofa")
    df = pd.DataFrame(ds['train'])
    os.makedirs(os.path.dirname(probe_file), exist_ok=True)
    df.to_csv(probe_file, index=False)
    print(f"Saved dataset to {probe_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--probe_file", type=str, default="data/sofa/SBIC-Pro.csv")
    parser.add_argument("--identity_file", type=str, default="data/sofa/identity_terms.json")
    parser.add_argument("--identity_dir", type=str, default="data/sofa/identity_terms")
    parser.add_argument("--output_file", type=str, default="data/sofa/SBIC-Pro-with-log-ratios.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.probe_file):
        download_sofa_dataset(args.probe_file)

    if not os.path.exists(args.identity_file):
        print(f"{args.identity_file} not found. Generating from CSVs in {args.identity_dir}...")
        generate_identity_terms_json(args.identity_dir, args.identity_file)

    analyze(
        model_name=args.model_name,
        probe_file=args.probe_file,
        identity_file=args.identity_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
    )
