import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import json

def load_model_and_tokenizer(model_id, device):
    if gptqmodel:
        from gptqmodel import GPTQModel
        model = GPTQModel.from_quantized(model_id, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def compute_perplexity(texts, tokenizer, model, batch_size=16, max_length=None, add_start_token=True):
    model.eval()
    results = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_tokenized_len = max_length - 1 if add_start_token and max_length else max_length

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=bool(max_tokenized_len),
            max_length=max_tokenized_len,
            add_special_tokens=False,
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        if add_start_token:
            bos = torch.full((input_ids.size(0), 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([bos, input_ids], dim=1)
            bos_mask = torch.ones_like(bos, dtype=attention_mask.dtype)
            attention_mask = torch.cat([bos_mask, attention_mask], dim=1)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        perplexity = torch.exp(loss)
        results.extend(perplexity.detach().cpu().tolist())

    return results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(args.model_id, device)

    df = load_dataset("copenlu/sofa", split="train").to_pandas()
    df["ppl"] = compute_perplexity(df["probe"].tolist(), tokenizer, model, args.batch_size, args.max_length)

    identity_ppl = {}
    for identity in tqdm(df["identity"].unique(), desc="Identity PPLs"):
        try:
            ppl = compute_perplexity([identity], tokenizer, model, batch_size=1, max_length=args.max_length)
            identity_ppl[identity] = ppl[0]
        except Exception:
            identity_ppl[identity] = np.nan

    df["norm"] = df["identity"].map(identity_ppl)
    df["ppl_star"] = df["ppl"] / df["norm"]
    df["log_ppl_star"] = np.log10(df["ppl_star"].replace(0, np.nan))

    grouped = df.dropna(subset=["log_ppl_star"]).groupby(["category", "stereotype"])["log_ppl_star"].var()
    sofa_scores = grouped.groupby("category").mean().to_dict()
    global_sofa_score = np.nanmean(list(sofa_scores.values()))

    print("\nSOFA Scores by Category (↓ best):")
    for cat, score in sofa_scores.items():
        print(f"{cat}: {score:.4f}")
    print(f"\nGlobal SOFA Score: {global_sofa_score:.4f}")

    model_name_part = args.model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    base = args.output if args.output else f"{model_name_part}_sofa"
    df.to_csv(f"{base}_results.csv", index=False)
    with open(f"{base}_metrics.json", "w") as f:
        json.dump({
            "model": args.model_id,
            "sofa_scores": sofa_scores,
            "global_sofa_score": global_sofa_score
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gptqmodel", action="store_true")
    args = parser.parse_args()
    main(args)