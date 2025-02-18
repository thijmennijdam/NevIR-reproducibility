import os
import pandas as pd
import argparse
import tqdm
from datasets import load_dataset
from evaluate.rerank_llms import calc_llms_rerankers
from transformers import pipeline
import torch

MODEL_MAP = {
    # "Qwen/Qwen2-1.5B-Instruct": ("Qwen/Qwen2-1.5B-Instruct", calc_llms_rerankers, "Qwen/Qwen2-1.5B-Instruct"),
    # "Qwen/Qwen2-7B-Instruct": ("Qwen/Qwen2-7B-Instruct", calc_llms_rerankers, "Qwen/Qwen2-7B-Instruct"),
    # "mistralai/Mistral-7B-Instruct-v0.3": ("mistralai/Mistral-7B-Instruct-v0.3", calc_llms_rerankers, "mistralai/Mistral-7B-Instruct-v0.3"),
    # "Llama-3.1-7B-instruct": ("meta-llama/Meta-Llama-3.1-8B-Instruct", calc_llms_rerankers, "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    # "Llama-3.2-3B-instruct": ("meta-llama/Llama-3.2-3B-Instruct", calc_llms_rerankers, "meta-llama/Llama-3.2-3B-Instruct"),
    "Finetuned Mistral": ("/scratch-shared/tnijdam_ir2/finetuned_1", calc_llms_rerankers, "/scratch-shared/tnijdam_ir2/finetuned_1")
}

ALL_MODELS = list(MODEL_MAP.values())

def predict(args):
    dataset = load_dataset("orionweller/NevIR", split="test")
    df = dataset.to_pandas()

    if args.models == ["all"]:
        selected_models = ALL_MODELS        
    else:
        selected_models = []
        for model in args.models:
            selected_models.append(MODEL_MAP[model])

    model_bar = tqdm.tqdm(selected_models, leave=True)
    for (model_name, model_func, model_details) in model_bar:
        print(f'model_func: {model_func}') 
        model_bar.set_description(f"Model {model_name}")
        model_results = []
        row_p_bar = tqdm.tqdm(df.iterrows())

        generator = pipeline("text-generation",
                            model=model_name,     
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            device_map='auto'
                            )

        i = 0
        for (row_idx, row) in row_p_bar:  # each instance

            if model_details is not None:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model_name=model_details,
                    generator=generator
                )
                model_results.append(results)
            else:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model=model,
                )
                model_results.append(results)

            i += 1
            if i == 100:
                break
            
        model_df = pd.DataFrame(model_results)
        
        print(model_df)
        model_df["pairwise_score"] = model_df.score.apply(lambda x: x == 1.0)
        overall_score = model_df.pairwise_score.mean()
        print(
            f'For model {model_name} the average score is {overall_score}'
        )

        if args.output is not None:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
                
            model_df.to_csv(os.path.join(args.output, f"results_{model_name}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        help="which split to evaluate on",
        type=str,
        default="validation"
    )
    parser.add_argument(
        "-m",
        "--models",
        help="name of the model to predict with or all models (default). Options include tfidf, colbertv1, dpr, or other dense sentencetransformers models",
        nargs="*",
        default=["all"],
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output directory to save results",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    predict(args)
