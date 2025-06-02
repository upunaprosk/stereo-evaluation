# Stereotype Evaluation in LLMs

This repository contains a few scripts for evaluating stereotypes and social bias in language models.  

Running scripts requires  `pip install transformers==4.51.3`. 
OPT implementation is broken in newer 4.52.x versions.


## StereoSet

StereoSet is a benchmark designed to evaluate stereotypical bias in language models across categories like gender, race, and profession using intrasentence format.
`stereoset.py` is based on [bias-bench](https://github.com/McGill-NLP/bias-bench) implementation. 

Models supported include: 
1) causal decoder LMs (OPT, LLaMA, Mistral + any model that can be loaded using `transformers.AutoModelForCausalLM`)
2) causal LMs quantized with GPTQModel: https://github.com/ModelCloud/GPTQModel
3) bert-like encoder models

### Usage

`
python stereoset.py --model_name_or_path "facebook/opt-125m" --persistent_dir "./"
`

### Arguments

| Argument               | Type   | Default              | Description |
|------------------------|--------|----------------------|-------------|
| `--persistent_dir`     | str    | `./`                | Directory where persistent data (input/output) will be stored. |
| `--file_name`          | str    | `test.json`          | Input file name for evaluation data. |
| `--model_name_or_path` | str    | `bert-base-uncased`  | HuggingFace model name or path to a pretrained checkpoint. |
| `--batch_size`         | int    | `1`                  | Batch size used during evaluation. |
| `--seed`               | int    | `None`               | Random seed used for reproducibility and experiment ID. |
| `--cache_dir`          | str    | `None`               | Directory for cached model files. |


## SoFA

`sofa.py` script runs the **SoFA (Social Fairness)** benchmark to evaluate social bias in language models.
Dataset: [copenlu/sofa](https://huggingface.co/datasets/copenlu/sofa)

### Usage

`python sofa.py --model_name "facebook/opt-125m"`

### Arguments

| Argument            | Type   | Default                                | Description |
|---------------------|--------|----------------------------------------|-------------|
| `--model_name`      | str    | `"gpt2"`                               | HF model name or path. |
| `--probe_file`      | str    | `"data/sofa/SBIC-Pro.csv"`             | Path to the CSV file containing probe sentences. |
| `--identity_file`   | str    | `"data/sofa/identities_by_category.json"` | JSON file with identity groups used in bias evaluation. |
| `--batch_size`      | int    | `512`                                  | Batch size for computing perplexities. |
| `--max_length`      | int    | `32`                                   | Maximum input sequence length. |
| `--gptqmodel`       | flag   | `False`                                | Use GPTQ quantized model if set. |

