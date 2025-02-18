import os
import pandas as pd
import argparse
import tqdm
from scipy.special import softmax
from datasets import load_dataset
from evaluate.rerank_rankgpt import calc_preffered_rankgpt_few_shot

MODEL_MAP = {
    "rankGPT-4o": ("rankGPT-4o", calc_preffered_rankgpt_few_shot, "gpt-4o"),
}

ALL_MODELS = list(MODEL_MAP.values())

def predict(args):
    dataset = load_dataset("orionweller/NevIR", split="test")
    df = dataset.to_pandas()
    train_df = pd.read_csv(args.nevir_train_path, sep="\t") 

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

        # 50 rows in df
        # df = df.head(50)
        row_p_bar = tqdm.tqdm(df.iterrows())
        model = None
        
        for (row_idx, row) in row_p_bar:  # each instance
            if model_details is not None:
                results, model, sim_score = model_func(
                    train_df,
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model_name=model_details,
                    model=model,
                    api_key=args.api_key,
                    x_shot=args.x_shot,
                )
                model_results.append(results)
            else:
                results, model, sim_score = model_func(
                    train_df,
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model=model,
                    api_key=args.api_key,
                    x_shot=args.x_shot,
                )
                model_results.append(results)

        model_df = pd.DataFrame(model_results)
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
    
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for OpenAI"
    )
    parser.add_argument(
        "--nevir_train_path",
        type=str,
        help="Path to the NevIR training data",
        default="NevIR_data/train_triplets.tsv"
    )
    parser.add_argument(
        "--x_shot",
        type=int,
        help="Number of examples to use for few-shot learning",
        default=1
    )
    
    args = parser.parse_args()
    predict(args)
