import os
import random
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from sentence_transformers import InputExample
from transformers import T5ForConditionalGeneration
from external.pygaggle.rerank.transformer import MonoT5
from finetune.monot5.pairwise_evaluator import PairwiseEvaluator
from finetune.monot5.accuracy_evaluator import AccuracyEvaluator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def convert_dataset_to_triplets(df, evaluation: bool = False):
    """
    Convert DataFrame into triplets for evaluation.
    """
    all_instances = []
    for _, row in df.iterrows():
        if evaluation:
            all_instances.append(
                InputExample(texts=[row["q1"], row["q2"], row["doc1"], row["doc2"]])
            )
    return all_instances

def eval_monoT5(checkpoint_path, excluIR_path):
    """
    Evaluate MonoT5 on NevIR and ExcluIR test sets for a given checkpoint.
    """
    _set_seed(42)

    # Load model and reranker
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(DEVICE)
    reranker = MonoT5(model=model)

    # Evaluate on ExcluIR
    excluIR_test_df = pd.read_csv(excluIR_path, sep="\t", header=None, names=["q1", "q2", "doc1", "doc2"])
    excluIR_test_data = convert_dataset_to_triplets(excluIR_test_df, evaluation=True)
    evaluator_excluIR = AccuracyEvaluator.from_input_examples(excluIR_test_data, name="test", model_name='monot5')
    excluIR_accuracy = evaluator_excluIR(model=reranker, output_path=None)

    # Evaluate on NevIR
    dataset_test = load_dataset("orionweller/NevIR", split="test")
    nevir_test_df = dataset_test.to_pandas()
    nevir_test_data = convert_dataset_to_triplets(nevir_test_df, evaluation=True)
    evaluator_nevir = PairwiseEvaluator.from_input_examples(nevir_test_data, name="test", model_name='monot5')
    nevir_accuracy = evaluator_nevir(model=reranker, output_path=None)

    return excluIR_accuracy, nevir_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MonoT5 model on ExcluIR and NevIR test sets")

    parser.add_argument('--checkpoint_base_dir', type=str, required=True, 
                        help='Base directory containing checkpoint-x directories')
    parser.add_argument('--excluIR_testdata', type=str, required=True, 
                        help='Path to the ExcluIR test data')
    parser.add_argument('--output_csv', type=str, required=True, 
                        help='Path to save the CSV with evaluation results')

    args = parser.parse_args()

    # Prepare result storage
    if os.path.exists(args.output_csv):
        # If the CSV already exists, load the existing results to avoid duplication
        results_df = pd.read_csv(args.output_csv)
        completed_epochs = set(results_df["epoch"])
    else:
        results_df = pd.DataFrame(columns=["epoch", "ExcluIR", "NevIR"])
        completed_epochs = set()

    # Get sorted list of checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(args.checkpoint_base_dir) if d.startswith("checkpoint-")]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(args.checkpoint_base_dir, checkpoint_dir)
        epoch = int(checkpoint_dir.split("-")[1])  # Extract epoch number

        # Skip if the epoch is already evaluated
        if epoch in completed_epochs:
            print(f"Skipping already evaluated checkpoint: {checkpoint_dir} (Epoch {epoch})")
            continue

        print(f"Evaluating checkpoint: {checkpoint_dir} (Epoch {epoch})")
        excluIR_acc, nevir_acc = eval_monoT5(checkpoint_path, args.excluIR_testdata)

        # Append result to the DataFrame
        results_df = pd.concat(
            [results_df, pd.DataFrame({"epoch": [epoch], "ExcluIR": [excluIR_acc], "NevIR": [nevir_acc]})],
            ignore_index=True,
        )

        # Write results to CSV after every epoch
        results_df.to_csv(args.output_csv, index=False)
        print(f"Epoch {epoch}: ExcluIR={excluIR_acc:.4f}, NevIR={nevir_acc:.4f}")
        print(f"Results written to {args.output_csv}")
