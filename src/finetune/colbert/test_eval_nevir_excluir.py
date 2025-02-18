import os
import random
import torch
import torch.nn as nn
import numpy as np
from transformers import set_seed
from colbert.modeling.colbert import ColBERT
from colbert_pairwise_evaluator import PairwiseEvaluator
from datasets import load_dataset
from sentence_transformers import InputExample
import argparse
import pandas as pd

def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def convert_dataset_to_triplets(df, evaluation: bool = False):
    """
    Convert a dataset into triplets for evaluation or training.

    Args:
        df: DataFrame containing the dataset.
        evaluation: If True, create evaluation triplets.

    Returns:
        List of InputExample instances.
    """
    all_instances = []
    for _, row in df.iterrows():
        if evaluation:
            all_instances.append(
                InputExample(texts=[row["q1"], row["q2"], row["doc1"], row["doc2"]])
            )
    print("Number of instances:", len(all_instances))
    return all_instances

def eval(check_point_path, output_path_nevir, output_path_excluIR, excluIR_path):
    """
    Evaluate the best checkpoint on the test set.

    Args:
        check_point_path (str): Path to the best checkpoint file.
        output_path (str): Path to save evaluation results.
    """
    _set_seed(42)

    print(f'Loading checkpoint from {check_point_path}')

    # Load ColBERT model
    colbert = ColBERT.from_pretrained(
        'bert-base-uncased',
        query_maxlen=32,
        doc_maxlen=180,
        dim=128,
        similarity_metric="l2",
        mask_punctuation=True
    )

    # Load checkpoint
    checkpoint = torch.load(check_point_path, map_location='cpu')
    try:
        colbert.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"[WARNING] Loading checkpoint with strict=False: {e}")
        colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    colbert = colbert.to('cuda' if torch.cuda.is_available() else 'cpu')
    colbert.eval()

    if output_path_excluIR:

        print("loading ExcluIR test data ...")
        excluIR_test_df = pd.read_csv(excluIR_path, sep="\t", header=None, names=["q1", "q2", "doc1", "doc2"])

        # Convert test data to triplets
        excl_test_data = convert_dataset_to_triplets(excluIR_test_df, evaluation=True)
        evaluator_test_excluIR = PairwiseEvaluator.from_input_examples(excl_test_data, name="test", model_name='colbert')

        print("Evaluating the model on the excluIR test set...")
        evaluation_accuracy_test = evaluator_test_excluIR(model=colbert, output_path=output_path_excluIR)
        print(f"ExlcuIR Test Accuracy: {evaluation_accuracy_test:.4f}")
        print(f"Saved to: {output_path_excluIR}")
    
    else:
        print('No evaluation on ExcluIR')

    if output_path_nevir:
        print("Loading NevIR test data...")
        dataset_test = load_dataset("orionweller/NevIR", split="test")
        test_df = dataset_test.to_pandas()

        # Convert test data to triplets
        test_data = convert_dataset_to_triplets(test_df, evaluation=True)
        evaluator_test = PairwiseEvaluator.from_input_examples(test_data, name="test", model_name='colbert')

        print("Evaluating the model on the NevIR test set...")
        evaluation_accuracy_test = evaluator_test(model=colbert, output_path=output_path_nevir)
        print(f"(NevIR) Test Accuracy: {evaluation_accuracy_test:.4f}")
        print(f"Saved to: {output_path_nevir}")

    else:
        print('No NevIR evaluation')

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Evaluate the best checkpoint on the test set")

    # Add arguments
    parser.add_argument('--check_point_path', type=str, required=True, 
                        help='Path to the best checkpoint file')
    parser.add_argument('--output_path_nevir', type=str, required=False, default=None,
                    help='Path to save evaluation results nevir (default: None)')
    parser.add_argument('--output_path_excluIR', type=str, required=False, default=None,
                        help='Path to save evaluation results excluIR')
    parser.add_argument('--excluIR_testdata', type=str, required=True, 
                        help='Path to save excluIR test data')
    

    # Parse arguments
    args = parser.parse_args()

    if args.output_path_nevir:
        os.makedirs(args.output_path_nevir, exist_ok=True)
    if args.output_path_excluIR:
        os.makedirs(args.output_path_excluIR, exist_ok=True)
        
    # Call the eval function with parsed arguments
    eval(check_point_path=args.check_point_path, output_path_nevir=args.output_path_nevir, output_path_excluIR=args.output_path_excluIR, excluIR_path= args.excluIR_testdata)