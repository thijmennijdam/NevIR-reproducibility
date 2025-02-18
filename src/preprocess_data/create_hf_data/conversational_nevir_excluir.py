from datasets import load_dataset, DatasetDict, concatenate_datasets
import os

# Names of the conversational datasets
dataset_names = ["thijmennijdam/ExcluIR-conversational", "thijmennijdam/NevIR-conversational"]
splits = ["train", "validation", "test"]

# Dictionary to store merged splits
merged_splits = {}

# Combine datasets for each split
for split_name in splits:
    datasets = []
    for dataset_name in dataset_names:
        # Load the specific split of the dataset
        dataset = load_dataset(dataset_name, split=split_name)
        datasets.append(dataset)
    
    # Concatenate all datasets for the current split
    merged_dataset = concatenate_datasets(datasets)
    merged_splits[split_name] = merged_dataset

# Create a DatasetDict from the merged splits
final_dataset = DatasetDict(merged_splits)

# Save the merged dataset locally
output_dir = "data/NevIR_ExcluIR/conversational"
os.makedirs(output_dir, exist_ok=True)
final_dataset.save_to_disk(output_dir)

print(f"Merged dataset successfully saved locally at: {output_dir}")

# Push the merged dataset to Hugging Face Hub
repo_name = "thijmennijdam/NevIR-ExcluIR-conversational"
print(f"Uploading the dataset to {repo_name}...")

# Upload to the Hugging Face Hub
final_dataset.push_to_hub(repo_name)

print(f"Merged dataset successfully uploaded to {repo_name}")
# Usage:
# uv run conversational_nevir_excluir.py