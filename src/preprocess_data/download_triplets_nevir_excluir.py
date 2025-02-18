import os
import pandas as pd

def create_directory(path: str):
    """Ensure the directory exists."""
    os.makedirs(path, exist_ok=True)

def download_dataset(dataset_name: str, output_dir: str):
    """
    Download a Hugging Face dataset and save each split locally as TSV files.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face.
        output_dir (str): The local directory to save the dataset.
    """
    from datasets import load_dataset

    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Ensure output directory exists
    create_directory(output_dir)

    # Save each split as a TSV file
    for split in dataset.keys():
        output_file = os.path.join(output_dir, f"{split}.tsv")
        df = dataset[split].to_pandas()
        df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved {split} split to {output_file}")

def create_merged_train_file(excluir_dir: str, nevir_dir: str):
    """
    Combine the train splits (TSV files) of ExcluIR and NevIR by literally
    stacking rows, preserving columns exactly as in the original files.
    """
    # Read the already-downloaded ExcluIR and NevIR train TSVs with pandas
    df_excluir = pd.read_csv(os.path.join(excluir_dir, "train.tsv"), sep="\t")
    df_nevir   = pd.read_csv(os.path.join(nevir_dir,  "train.tsv"), sep="\t")

    # Ensure columns match (optional safeguard)
    if list(df_excluir.columns) != list(df_nevir.columns):
        raise ValueError("Train files have different columns; cannot combine safely.")

    # Concatenate row-wise
    merged_df = pd.concat([df_excluir, df_nevir], ignore_index=True)

    # Create output dir and save the merged file
    output_dir = "data/NevIR_ExcluIR"
    create_directory(output_dir)
    output_file = os.path.join(output_dir, "train.tsv")

    # Save exactly the same way as the original:
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved merged train split to {output_file}")

if __name__ == "__main__":
    # Directories to save ExcluIR and NevIR data
    excluir_dir = "data/ExcluIR"
    nevir_dir   = "data/NevIR"

    # Download and save ExcluIR data
    print("Downloading ExcluIR_triplets dataset...")
    download_dataset("thijmennijdam/ExcluIR_triplets", excluir_dir)

    # Download and save NevIR data
    print("\nDownloading NevIR_triplets dataset...")
    download_dataset("thijmennijdam/NevIR_triplets", nevir_dir)

    print("\nDownload completed!")

    # Create a merged train file
    print("\nCreating merged train file...")
    create_merged_train_file(excluir_dir, nevir_dir)
