import os
import random
import torch

from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message

from datasets import load_dataset

from sentence_transformers import (
    InputExample
)
import os
import torch
import numpy as np
import argparse
from datasets import load_dataset

# grab locally
from finetune.colbert.colbert_pairwise_evaluator import PairwiseEvaluator


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
    
    print("DELETE LATER, ONLY DOING 10 INSTANCES")
    # only do first 20 instances
    all_instances = all_instances[:10]
    return all_instances

def eval(num_epochs, check_point_path, output_path):
    _set_seed(42)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # # XXX sien added this outer epoch loop
    # num_epochs = 20
    # check_point_path = '/path to check_point'

    for epoch in range(num_epochs):
        # /home/scur0987/project/ColBERT/colbert_experiment_train/NevIR/train.py/msmarco.psg.l2/checkpoints/colbert-0.dnn
        print(f'checkpoint = {check_point_path}colbert-{epoch}.dnn')



        colbert = ColBERT.from_pretrained('bert-base-uncased',
                                            query_maxlen=32,
                                            doc_maxlen=180,
                                            dim=128,
                                            similarity_metric="l2",
                                            mask_punctuation=True)

        if check_point_path is not None:
            checkpoint = torch.load(f'{check_point_path}colbert-{epoch}.dnn', map_location='cpu')

            try:
                colbert.load_state_dict(checkpoint['model_state_dict'])
            except:
                print_message("[WARNING] Loading checkpoint with strict=False")
                colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print('provide check_point_path')
            break

        colbert = colbert.to(DEVICE)

        print("Loading data...")

        dataset_test = load_dataset("orionweller/NevIR", split="test")
        test_df = dataset_test.to_pandas()

        dataset_validation = load_dataset("orionweller/NevIR", split="validation")
        validation_df = dataset_validation.to_pandas()

        print("Loading evaluation...")
        validation_data = convert_dataset_to_triplets(validation_df, evaluation=True) # NevIR to triplets validation
        evaluator = PairwiseEvaluator.from_input_examples(validation_data, name="validation", model_name='colbert') # triplet evaluator

        # test_data = convert_dataset_to_triplets(test_df, evaluation=True)
        # evaluator_test = PairwiseEvaluator.from_input_examples(test_data, name="test", model_name='colbert')

                    
        # Evaluate the model using PairwiseEvaluator after completing the epoch
        colbert.eval()
        print(f"\nEvaluating the model after epoch {epoch + 1}/{num_epochs}")
        evaluation_accuracy = evaluator(model=colbert, output_path=output_path, epoch=epoch + 1)
        print(f"Evaluation Accuracy after epoch {epoch + 1}: {evaluation_accuracy:.4f}\n")

        # evaluation_accuracy_test = evaluator_test(model=colbert, epoch=epoch + 1)
        # print(f"Evaluation test Accuracy after epoch {epoch + 1}: {evaluation_accuracy:.4f}\n")


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

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    eval(num_epochs=args.num_epochs, check_point_path=args.check_point_path, output_path = args.output_path)



