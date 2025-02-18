# Official Repository of "Reproducing NevIR: Negation in Neural Information Retrieval"

This repository provides all the resources needed to reproduce the models evaluated in our paper. To get started, follow the instructions in the **"Evaluate on NevIR"** section. Note that some models require permission from the owner on Hugging Face before use.

For reproducing our fine-tuning experiments, refer to the **"Finetune experiments"** section.

To manage different virtual environments efficiently, we highly recommend using uv: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/). It is significantly faster than pip (10-100x) and allows easy creation of virtual environments with different Python versions, which are required for this project.

This repository is organized as follows:

```
src  
├── evaluate/      # Contains scripts for evaluating models, primarily on the NevIR dataset.  
├── finetune/      # Includes code for finetuning models and evaluating them on various development datasets (e.g., NevIR, MS MARCO).  
├── external/      # Holds external repositories such as ColBERT, pygaggle, MTEB, and RankGPT used in the project.  

requirements/      # Contains requirement files needed to set up different environments.  
models/            # Contains model checkpoints and weights used during training and evaluation.  
```

# Evaluate on NevIR  

### **Dense Retrieval Models**  

To evaluate the following models:  

- `cross-encoder/qnli-electra-base`  
- `cross-encoder/stsb-roberta-large`  
- `cross-encoder/nli-deberta-v3-base`  
- `msmarco-bert-base-dot-v5`  
- `multi-qa-mpnet-base-dot-v1`  
- `DPR`  
- `msmarco-bert-co-condensor`  
- `DRAGON`  

Run the following commands:  

```bash
uv venv .envs/dense_env --python 3.10
source .envs/dense_env/bin/activate
uv pip install -r requirements/requirements_dense.txt
uv pip install -e .
uv run python src/evaluate/evaluate_dense.py
```


### **Sparse Retrieval Models**  

To evaluate the following models:  

- `TF-IDF`  
- `SPLADEv2 ensemble-distill`  
- `SPLADEv2 self-distill`  
- `SPLADEv3`  

Run the following commands:  

```bash
uv venv .envs/sparse_env --python 3.10
source .envs/sparse_env/bin/activate
uv pip install -r requirements/requirements_sparse.txt
uv pip install transformers==4.29.0
uv pip install -e .
uv run python src/evaluate/evaluate_sparse.py
```  

### **Reranking Models**  

To evaluate the following models:  

- `MonoT5 small` (Nogueira et al., 2020)  
- `MonoT5 base (default)` (Nogueira et al., 2020)  
- `MonoT5 large` (Nogueira et al., 2020)  
- `MonoT5 3B` (Nogueira et al., 2020)  

Run the following commands:  

```bash
module load 2023
module load Anaconda3/2023.07-2
conda env create -f requirements/rerank.yml
source activate rerank
pip install -r requirements/requirements_rerank.txt
pip uninstall spacy thinc pydantic
pip install spacy thinc pydantic
pip install -e .
python src/evaluate/evaluate_rerankers.py
```  

### **ColBERT Models**  

To evaluate the following models:  

- `ColBERTv1`  
- `ColBERTv2`  

#### **Download ColBERT Weights**  

```bash
mkdir -p ./models/colbert_weights
chmod +x ./requirements/install_git_lfs.sh
./requirements/install_git_lfs.sh
git clone https://huggingface.co/orionweller/ColBERTv1 models/colbert_weights/ColBERTv1
git clone https://huggingface.co/colbert-ir/colbertv2.0 models/colbert_weights/colbertv2.0
```  

#### **Run Evaluation**  

```bash
uv venv .envs/colbert_env --python 3.10
source .envs/colbert_env/bin/activate
uv pip install -r requirements/requirements_colbert.txt
uv pip install src/external/ColBERT
uv pip install -e .
uv run python src/evaluate/evaluate_colbert.py
```

# **LLM-Based Models**  

### **Bi-Encoders**  

To evaluate the following models:  

- `GritLM-7B`  
- `RepLlama`  
- `promptriever-llama3.1-8b-v1`  
- `promptriever-mistral-v0.1-7b-v1`  
- `OpenAI text-embedding-3-small`  
- `OpenAI text-embedding-3-large`  
- `gte-Qwen2-1.5B-instruct`  
- `gte-Qwen2-7B-instruct`  

We use a fork of the [MTEB GitHub repository](https://github.com/thijmennijdam/mteb) that includes bug fixes, custom rerankers, and additional code for running these models. First, clone this fork:  

```bash
git clone --branch old_version_v2 https://github.com/thijmennijdam/mteb.git src/external/mteb
cd src/external/mteb
```

Now, create the environment and install the required dependencies:  

```bash
uv sync
source .venv/bin/activate
uv pip install 'mteb[peft]'
uv pip install 'mteb[jina]'
uv pip install 'mteb[flagembedding]'
uv pip install gritlm
uv pip install --upgrade setuptools
uv pip uninstall triton
uv pip install triton
uv pip install -e ./../../..
```

For these models, you need to install `huggingface` and log in:  

```bash
uv pip install huggingface_hub
huggingface-cli login
```

Evaluate the bi-encoders using the following commands (**Note**: Set the environment variable `OPENAI_API_KEY` before running OpenAI models):  

```bash
uv run python eval_nevir.py --model "castorini/repllama-v1-7b-lora-passage"
uv run python eval_nevir.py --model "GritLM/GritLM-7B"
uv run python eval_nevir.py --model "text-embedding-3-small"
uv run python eval_nevir.py --model "text-embedding-3-large"
uv run python eval_nevir.py --model "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
uv run python eval_nevir.py --model "Alibaba-NLP/gte-Qwen2-7B-instruct"
uv run python eval_nevir.py --model "samaya-ai/promptriever-llama2-7b-v1"
uv run python eval_nevir.py --model "samaya-ai/promptriever-mistral-v0.1-7b-v1"
```

---

### **Rerankers**  

To evaluate the following models:  

- `bge-reranker-base`  
- `bge-reranker-v2-m3`  
- `jina-reranker-v2-base-multilingual`  

Ensure you provide a path to the bi-encoder results using `--previous_results`. Example:  

```bash
uv run python eval_nevir.py --model "jinaai/jina-reranker-v2-base-multilingual" --previous_results "path/to/results"
uv run python eval_nevir.py --model "BAAI/bge-reranker-base" --previous_results "path/to/results"
uv run python eval_nevir.py --model "BAAI/bge-reranker-v2-m3" --previous_results "path/to/results"
```

#### **RankLlama**  
The `RankLlama` reranker is not yet integrated into MTEB, so use a separate script:  

```bash
uv run python ../../evaluate/evaluate_rankllama.py
```

---

### **RankGPT Models**  

To evaluate the following models:  

- `RankGPT GPT-4o-mini`  
- `RankGPT GPT-4o`  
- `RankGPT o3-mini`  

Use any environment, install `openai`, and provide your OpenAI API key:  

```bash
uv pip install openai==1.56.1
uv run python src/evaluate/evaluate_rankgpt.py --api_key "your_api_key"
```

---

### **Listwise LLM Re-rankers**  

To evaluate the following models:  

- `Qwen2-1.5B-Instruct`  
- `Qwen2-7B-Instruct`  
- `Llama-3.2-3B-Instruct`  
- `Llama-3.1-7B-Instruct`  
- `Mistral-7B-Instruct-v0.3`  

Run the following commands:  

```bash
uv venv .envs/llm_env --python 3.10
source .envs/llm_env/bin/activate
uv pip install -r requirements/requirements_llms.txt 
uv pip install -e .
uv run python src/evaluate/evaluate_llms.py 
```

# **Fine-Tuning Experiments**  

This section describes how to reproduce the fine-tuning experiments. First, download the necessary data. Use any of the provided environments and ensure `gdown`, `polars`, and `sentence_transformers` are installed:  

```bash
uv pip install gdown sentence_transformers polars
uv run python src/preprocess_data/data.py
```

The following datasets will be installed:  

- NevIR training, validation, and test triplets (TSV format)  
- MS MARCO collection and top 1000 documents, including a custom file with 500 queries  
- ExcluIR dataset  
- Merged dataset of MS MARCO and NevIR (stored in `merged_dataset/`)  
- Merged dataset of NevIR and ExcluIR (stored in `Exclu_NevIR_data/`)  

### **Experiment Configuration**  

The following sections provide example commands for fine-tuning on NevIR and evaluating these checkpoints on both NevIR and MS MARCO. To run different experiments, modify the dataset paths in the arguments.  

For example, fine-tuning on NevIR:  

```bash
--triples data/NevIR_data/train_triplets.tsv
```  

Fine-tuning on ExcluIR:  

```bash
--triples data/ExcluIR_data/train_samples.tsv
```

---

## **ColBERTv1**  

Activate the ColBERT environment:  

```bash
source .envs/colbert_env/bin/activate
```

Fine-tune on NevIR:  

```bash
uv run python src/finetune/colbert/finetune_colbert.py \
--amp \
--doc_maxlen 180 \
--mask-punctuation \
--bsize 32 \
--accum 1 \
--triples data/NevIR_data/train_triplets.tsv \
--root models/checkpoints/colbert/finetune_nevir \
--experiment NevIR \
--similarity l2 \
--run nevir \
--lr 3e-06 \
--checkpoint models/colbert_weights/ColBERTv1/colbert-v1.dnn
```

Modify `--triples` to fine-tune on a different dataset. Use `--root` to specify a different checkpoint storage directory. **Do not modify `--checkpoint`, as it provides the starting weights.**  

### **Evaluation**  

Evaluate each checkpoint on **NevIR**:  

```bash
uv run python src/finetune/colbert/dev_eval_nevir.py \
--num_epochs 20 \
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/ \
--output_path results/colbert/finetune_nevir/NevIR_performance
```

Evaluate on **MS MARCO**:  

```bash
yes | uv run python src/finetune/colbert/dev_eval_msmarco.py \
--num_epochs 20 \
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/ \
--output_path results/colbert/finetune_nevir/MSMarco_performance \
--experiment NevIR_ft
```

Modify `--check_point_path` and `--output_path` based on the experiment.

### **Final Evaluation on Test Sets**  

To evaluate the best model on **NevIR** and **ExcluIR**, run:  

```bash
yes | uv run python src/finetune/colbert/test_eval_nevir_excluir.py \ 
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/colbert-19.dnn \
--output_path_nevir results/colbert/finetune_nevir/best_model_performance/NevIR \
--output_path_excluIR results/colbert/finetune_nevir/best_model_performance/excluIR \
--excluIR_testdata data/ExcluIR_data/excluIR_test.tsv 
```

---

## **MultiQA-mpnet-base**  

The **ColBERT** environment can be reused:  

```bash
source .envs/colbert_env/bin/activate
```

Fine-tune on NevIR and evaluate on NevIR and MS MARCO after each epoch:  

```bash
uv run python src/finetune/multiqa/nevir_finetune.py
```

Fine-tune on the merged dataset:  

```bash
uv run python src/finetune/multiqa/merged_finetune.py
```

### **Final Evaluation on Test Sets**  

Evaluate the best model on ExcluIR and NevIR:  

```bash
uv run python src/finetune/multiqa/evaluate_excluIR_multiqa.py
uv run python src/finetune/multiqa/evaluate_nevIR_multiqa.py
```

---

## **MonoT5-base-MS MARCO-10k**  

A separate environment is required (Python 3.8 is needed):  

```bash
uv venv .envs/finetune_monot5_env --python 3.8
source .envs/finetune_monot5_env/bin/activate
uv pip install -r requirements/requirements_finetune_monot5.txt
```

Fine-tune on NevIR:  

```bash
uv run python src/finetune/monot5/finetune_monot5.py
```

### **Evaluation**  

Switch back to the rerank environment:  

```bash
module load 2023 && module load Anaconda3/2023.07-2 && source activate rerank
```

Evaluate on NevIR:  

```bash
python src/finetune/monot5/dev_eval_nevir.py
```

Evaluate on MS MARCO:  

```bash
python src/finetune/monot5/dev_eval_msmarco.py \
--model_name_or_path models/checkpoints/monot5/finetune_nevir/checkpoint-29 \
--initial_run data/MSmarco_data/top1000.txt \
--corpus data/MSmarco_data/collection.tsv \
--queries data/MSmarco_data/queries.dev.small.tsv \
--output_run results/monot5/eval_msmarco \
--qrels data/MSmarco_data/qrels.dev.small.tsv
```  