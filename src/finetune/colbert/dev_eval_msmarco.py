import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import sys

from sentence_transformers import (
    InputExample
)
import os
import torch
import numpy as np
import argparse

# grab locally
from external.ColBERT.colbert.test_epoch import main_test

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


def call_main_test(check_point, epoch, experiment, output_path):
    # Simulate the command-line arguments

    print(f'check point name = {check_point}')
    sys.argv = [
        'main_test.py',  # Name of the script (simulated)
        '--amp',
        '--doc_maxlen', '180',
        '--mask-punctuation',
        '--collection', 'data/MSmarco_data/collection.tsv',
        '--queries', 'data/MSmarco_data/queries.dev.small.tsv',
        '--topk', 'data/MSmarco_data/500top1000.dev', # pAS AAN
        '--checkpoint', check_point,
         '--root', output_path,
        # '--root', 'ColBERT/colbert_experiment_eval',
        '--experiment', f'{experiment}_{epoch}',
        '--qrels', 'data/MSmarco_data/qrels.dev.small.tsv'
    ]
    
    # Call the main_test function
    main_test(epoch)



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
    
    print("DELETE LATER, TAKING ONLY 10 INSTANCES")
    all_instances = all_instances[:10]
    return all_instances

def eval(num_epochs, check_point_path, experiment, output_path):

    _set_seed(42)

    for epoch in range(num_epochs):
        print(f'checkpoint = {check_point_path}colbert-{epoch}.dnn')
        call_main_test(check_point = f'{check_point_path}colbert-{epoch}.dnn', epoch=epoch, experiment=experiment, output_path=output_path)
    


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Script for training a model")

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--check_point_path', type=str, required=True, 
                        help='Path to the checkpoint file')

    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to output')                 

    parser.add_argument('--experiment', type=str, required=True, 
                        help='Name of experiment')   
    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    eval(num_epochs=args.num_epochs, check_point_path=args.check_point_path, experiment=args.experiment, output_path = args.output_path)

