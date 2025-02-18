from datasets import load_dataset, Dataset
import pandas as pd
import os
import argparse


def create_directory(output_path: str):
    """Ensure the directory for the given path exists."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def prepare_excluir_triplets(dataset_name: str, output_dir: str, hf_account: str):
    """
    Load the ExcluIR dataset and create triplets using q2 as query, q2 as positive, and q1 as negative.
    Then upload the dataset to Hugging Face.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Function to generate triplets for a split
    def generate_triplets(split):
        rows = []
        for row in dataset[split]:
            rows.append({
                "query": row["q2"],
                "positive": row["doc2"],
                "negative": row["doc1"]
            })
        return pd.DataFrame(rows)

    # Prepare output paths
    train_output = os.path.join(output_dir, "excluir_train_triplets.tsv")
    val_output = os.path.join(output_dir, "excluir_val_triplets.tsv")
    test_output = os.path.join(output_dir, "excluir_test_triplets.tsv")

    # Create triplets for train, validation, and test sets
    train_triplets = generate_triplets("train")
    val_triplets = generate_triplets("validation")
    test_triplets = generate_triplets("test")

    # Save triplets to TSV files
    create_directory(train_output)
    train_triplets.to_csv(train_output, sep='\t', index=False, header=False)

    create_directory(val_output)
    val_triplets.to_csv(val_output, sep='\t', index=False, header=False)

    create_directory(test_output)
    test_triplets.to_csv(test_output, sep='\t', index=False, header=False)

    print(f"Triplets saved locally in {output_dir}")

    # Convert to Hugging Face Datasets and push to the Hub
    for split, df in [("train", train_triplets), ("validation", val_triplets), ("test", test_triplets)]:
        df.replace("\n", " ", regex=True, inplace=True)
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f"{hf_account}/ExcluIR_triplets", split=split)
        print(f"Uploaded {split} dataset to {hf_account}/ExcluIR_triplets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and upload ExcluIR triplets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ExcluIR",
        help="Directory to save the triplets. Default is 'data/ExcluIR'."
    )
    
    parser.add_argument(
        "--hf_account",
        type=str,
        default="thijmennijdam",
        help="Hugging Face account name. Default is 'thijmennijdam'."
    )
    
    args = parser.parse_args()

    prepare_excluir_triplets(
        dataset_name="thijmennijdam/ExcluIR",
        output_dir=args.output_dir,
        hf_account=args.hf_account
    )


# Usage:
# uv run python src/preprocess_data/create_hf_data/triplets_excluir.py 
