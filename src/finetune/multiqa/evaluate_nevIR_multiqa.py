import os
import json
import pandas as pd
import argparse
import ast
import numpy as np
import tqdm
from collections import defaultdict
from scipy.special import softmax
import torch
from datasets import load_dataset

# NOTE: if you are adding a new type of model not covered by these 
# frameworks you'll want to add and import it here
from finetune_calc_dense import fine_calc_preferred_dense
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    # np.random.seed(seed)  # Set seed for NumPy
    # random.seed(seed)  # Set seed for Python's random module
    torch.manual_seed(seed)  # Set seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch on current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for PyTorch on all GPUs

    # Allow non-deterministic behavior for performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def predict(args):
    dataset = load_dataset("orionweller/NevIR", split="test")

    df = dataset.to_pandas()

    selected_models = [("multi-qa-mpnet-base-dot-v1", fine_calc_preferred_dense, "multi-qa-mpnet-base-dot-v1")]

    model_bar = tqdm.tqdm(selected_models, leave=True)
    for (model_name, model_func, model_details) in model_bar:
        print(f'model_func: {model_func}') 
        model_bar.set_description(f"base_model {model_name}")
        model_results = []
        row_p_bar = tqdm.tqdm(df.iterrows())
        model = None
        i = 0
        for (row_idx, row) in row_p_bar:  # each instance
            # if i > 0 :
            #     break
            # i += 1
            
            if model_details is not None:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    args.path_to_finetuned_model, 
                    args.load_finetune, 
                    base_model="multi-qa-mpnet-base-dot-v1",
                )
                model_results.append(results)

        model_df = pd.DataFrame(model_results)
        model_df["q1_probs"] = model_df.q1.apply(lambda x: softmax(x))
        model_df["q2_probs"] = model_df.q2.apply(lambda x: softmax(x))
        model_df["pairwise_score"] = model_df.score.apply(lambda x: x == 1.0)
        overall_score = model_df.pairwise_score.mean()
        print(
            f'For model {model_name} the average score is {overall_score}'
        )

        if args.output is not None:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            
            if str2bool(args.load_finetune):
                finetune_set = args.path_to_finetuned_model.split('/')[-3]
                model_df.to_csv(os.path.join(args.output, f"results_{model_name}_{finetune_set}.csv"))
            else:
                model_df.to_csv(os.path.join(args.output, f"results_{model_name}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        help="output directory to save results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-se",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-ptf",
        "--path_to_finetuned_model",
        type=str,
        default="/home/scur0987/project/IR2/results_finetune/finetune_NevIR/sentence-transformers_multi-qa-mpnet-base-dot-v1/model",
        help="Path to the finetuned model",
    )
    parser.add_argument(
        "-lf",
        "--load_finetune",
        type=bool,
        default=True,
        help="Load the finetuned model",
    )
    args = parser.parse_args()
    _set_seed(args.seed)
    ptf = args.path_to_finetuned_model
    print("---Testing---")
    print("testing on nevIR dataset")
    if str2bool(args.load_finetune):
        print("loading finetuned model")
        print(ptf)
    else:
        print("not loading finetuned model")
    predict(args)

# example usage:
# python evaluate_nevIR.py -o test_nevir_results_finetune -se 42 -ptf /home/scur0987/project/IR2/results_finetune/finetune_NevIR/sentence-transformers_multi-qa-mpnet-base-dot-v1/model -lf True
