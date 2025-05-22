import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import torch
import sys
from logbar import LogBar


class Perplexity:
    def __init__(self, model, tokenizer, texts, n_ctx=512):
        self._model = model
        self._tokenizer = tokenizer
        self._text = self._prepare_text(texts)
        self._n_ctx = n_ctx

    def _prepare_text(self, texts):
        joined = "\n".join([t.strip() for t in texts if isinstance(t, str) and len(t.strip()) > 0])
        return joined

    @staticmethod
    def softmax(logits):
        e_x = torch.exp(logits - torch.max(logits))
        return e_x / torch.sum(e_x, dim=0)

    def calculate(self, n_batch=1024):
        self._tokenizer.model_max_length = sys.maxsize
        tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)

        nll = 0.0
        count = 0
        all_perplexity = []

        with LogBar.shared().pb(range(len(tokens[0]) // self._n_ctx)).title("Perplexity: - ").manual() as pb:
            for i in pb:
                nll, count = self._process_batch(i, self._n_ctx, n_batch, tokens, nll, count)
                curr_ppl = np.exp(nll / count)
                all_perplexity.append(curr_ppl)
                pb.title(f"Perplexity: {curr_ppl:.4f}").draw()

        return all_perplexity

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        start = i * n_ctx
        end = start + n_ctx
        num_batches = (n_ctx + n_batch - 1) // n_batch
        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)
            token_org = tokens[0][batch_start].item()

            if j == 0 and self._tokenizer.bos_token_id is not None:
                tokens[0][batch_start] = self._tokenizer.bos_token_id

            with torch.no_grad():
                out = self._model(tokens[:, batch_start: batch_start + batch_size])
            tokens[0][batch_start] = token_org

            logits.append(out.logits.detach())

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j]
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]
            nll += -torch.log(torch.where(prob > 0, prob, torch.tensor(1e-8))).item()
            count += 1

        return nll, count

def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: add int4 precision
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def compute_probe_ppls(df, model, tokenizer):
    probes = df['probe'].dropna().tolist()
    ppl = Perplexity(model, tokenizer, probes)
    scores = ppl.calculate()
    df['ppl_probe'] = scores[:len(df)]
    return df


def compute_identity_ppls(identity_file, model, tokenizer):
    with open(identity_file, "r") as f:
        identity_dict = json.load(f)

    results = {}
    for group, terms in identity_dict.items():
        filtered = [t.strip().capitalize() for t in terms if isinstance(t, str) and len(t.strip().split()) > 1]
        if not filtered:
            results[group] = np.nan
            continue
        ppl = Perplexity(model, tokenizer, filtered)
        scores = ppl.calculate()
        results[group] = np.mean(scores)
    return results


def compute_bias_score(df, identity_ppls):
    scores = []
    for _, row in df.iterrows():
        cat = row['category']
        ppl_identity = identity_ppls.get(cat, np.nan)
        ppl_probe = row['ppl_probe']
        if np.isnan(ppl_identity) or ppl_identity <= 0 or ppl_probe <= 0:
            scores.append(np.nan)
        else:
            scores.append(np.log10(ppl_probe / ppl_identity))
    df['bias_score_log10'] = scores
    return df


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


def download_sofa_dataset(probe_file):
    ds = load_dataset("copenlu/sofa")
    df = pd.DataFrame(ds["train"])
    os.makedirs(os.path.dirname(probe_file), exist_ok=True)
    df.to_csv(probe_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--probe_file", type=str, default="data/sofa/SBIC-Pro.csv")
    parser.add_argument("--identity_file", type=str, default="data/sofa/identity_terms.json")
    parser.add_argument("--identity_dir", type=str, default="data/sofa/identity_terms")
    parser.add_argument("--output_file", type=str, default="data/sofa/SBIC-Pro-with-bias.csv")
    args = parser.parse_args()

    if not os.path.exists(args.probe_file):
        download_sofa_dataset(args.probe_file)

    if not os.path.exists(args.identity_file):
        generate_identity_terms_json(args.identity_dir, args.identity_file)

    df = pd.read_csv(args.probe_file)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    df = compute_probe_ppls(df, model, tokenizer)
    identity_ppls = compute_identity_ppls(args.identity_file, model, tokenizer)
    df = compute_bias_score(df, identity_ppls)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()