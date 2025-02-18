import os
import random
import torch

from datasets import load_dataset
from sentence_transformers import (
    InputExample
)
import os
import torch
import numpy as np
import argparse
from datasets import load_dataset

from finetune.monot5.pairwise_evaluator import PairwiseEvaluator
from transformers import T5ForConditionalGeneration
from external.pygaggle.rerank.transformer import MonoT5
from tqdm import tqdm

DEVICE = torch.device('cuda')

#TODO: move to utils
def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    # np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module
    torch.manual_seed(seed)  # Set seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch on current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for PyTorch on all GPUs

    # Allow non-deterministic behavior for performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def convert_dataset_to_triplets(df, evaluation: bool = False):
    all_instances = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if not evaluation:
            all_instances.extend([
                InputExample(texts=[row['q1'], row['doc1']], label=1),
                InputExample(texts=[row['q1'], row['doc2']], label=0),
                InputExample(texts=[row['q2'], row['doc2']], label=1),
                InputExample(texts=[row['q2'], row['doc1']], label=0),
            ])
        else:
            all_instances.extend([
                InputExample(texts=[row["q1"], row["q2"], row["doc1"], row["doc2"]]),
            ])
    print("Number of instances:", len(all_instances))
    return all_instances

def rename_checkpoints(checkpoint_dir):
    # List all items in the directory
    items = os.listdir(checkpoint_dir)
    
    # Filter to only include directories starting with "checkpoint-"
    checkpoint_dirs = []
    for item in items:
        full_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(full_path) and item.startswith("checkpoint-"):
            # Extract the number after the dash
            parts = item.split('-')
            if len(parts) == 2 and parts[1].isdigit():
                checkpoint_number = int(parts[1])
                checkpoint_dirs.append((checkpoint_number, full_path))
    
    # Sort by the numeric value
    checkpoint_dirs.sort(key=lambda x: x[0])
    
        # Rename directories to have continuous numbering
    for i, (_, old_path) in enumerate(checkpoint_dirs, start=1):
        new_name = f"checkpoint-{i}"
        new_path = os.path.join(checkpoint_dir, new_name)
        os.rename(old_path, new_path)
    
def eval(num_epochs, check_point_path, output_path):
    _set_seed(42)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for epoch in tqdm(range(1, 21)):
        rename_checkpoints(args.check_point_path)

        epoch_check_point_path = f'{check_point_path}/checkpoint-{epoch}'
        print("LOADING MODEL FROM CHECKPOINT")
        print(f'checkpoint = {epoch_check_point_path}')
        
        model = T5ForConditionalGeneration.from_pretrained(epoch_check_point_path)
        reranker = MonoT5(model=model)

        print("Loading data...")

        dataset_test = load_dataset("orionweller/NevIR", split="test")
        test_df = dataset_test.to_pandas()

        dataset_validation = load_dataset("orionweller/NevIR", split="validation")
        validation_df = dataset_validation.to_pandas()

        print("Loading evaluation...")
        validation_data = convert_dataset_to_triplets(validation_df, evaluation=True) # NevIR to triplets validation
        evaluator = PairwiseEvaluator.from_input_examples(validation_data, name="validation", model_name='monot5') # triplet evaluator

        # Evaluate the model using PairwiseEvaluator after completing the epoch
        print(f"\nEvaluating the model after epoch {epoch}/{num_epochs}")
        evaluation_accuracy = evaluator(model=reranker, output_path=output_path, epoch=epoch)
        print(f"Evaluation Accuracy after epoch {epoch}: {evaluation_accuracy:.4f}\n")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Script for training a model")

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--check_point_path', type=str, required=False, default="models/checkpoints/monot5/finetune_nevir",
                        help='Path to the checkpoint file')

    parser.add_argument('--output_path', type=str, required=False, default="results/monot5",
                        help='Path to output')                 

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    eval(num_epochs=args.num_epochs, check_point_path=args.check_point_path, output_path = args.output_path)



