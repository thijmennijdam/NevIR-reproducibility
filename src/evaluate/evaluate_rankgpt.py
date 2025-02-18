import os
import pandas as pd
import argparse
import tqdm
from scipy.special import softmax
from datasets import load_dataset
from evaluate.rerank_rankgpt import calc_preffered_rankgpt

MODEL_MAP = {
    "rankGPT-4o-mini": ("rankGPT-4o-mini", calc_preffered_rankgpt, "gpt-4o-mini"),
    "rankGPT-4o": ("rankGPT-4o", calc_preffered_rankgpt, "gpt-4o"),
    'rankGPT-o3-mini': ("rankGPT-o3-mini", calc_preffered_rankgpt, "o3-mini"),
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
        
        # df = df.head(50)
        # start from 50 onwards
        # df = df.iloc[50:]
        row_p_bar = tqdm.tqdm(df.iterrows())
        model = None
        # Define output file
        output_path = os.path.join(args.output_dir, f"results_{model_name}.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Ensure header is written only once
        write_header = not os.path.exists(output_path)
        
        for (row_idx, row) in row_p_bar:  # each instance
            
            if model_details is not None:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model_name=model_details,
                    model=model,
                    api_key=args.api_key
                    
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

            # Convert result to DataFrame with explicit index
            result_df = pd.DataFrame([results])
            result_df["pairwise_score"] = result_df["score"].apply(lambda x: x == 1.0)
            result_df.index = [row_idx]  # Explicitly set the index

            # Append new row to CSV file with index
            result_df.to_csv(output_path, mode='a', index=True, header=write_header)
            write_header = False  # Ensure header is only written once

        # Compute and print the overall score efficiently
        model_df = pd.read_csv(output_path, index_col=0)  # Read with index
        overall_score = model_df["pairwise_score"].mean()
        print(f'For model {model_name} the average score is {overall_score}')

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
    
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for OpenAI"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="output directory to save results",
        default="results/evaluate_nevir"
    )
    args = parser.parse_args()
    predict(args)
