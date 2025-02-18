from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import HfApi
import json
import os
import textwrap

# Step 1: Load the dataset from Hugging Face
dataset_name = "thijmennijdam/ExcluIR"

splits = ["train", "validation", "test"]  # All splits
converted_splits = {}

def create_entry(q2, doc1, doc2, swap=False):
    if swap:
        # Swap the documents
        doc1, doc2 = doc2, doc1
        response = "[1] > [2]"
    else:
        response = "[2] > [1]"
    
    conversation = [
        {
            "from": "system",
            "value": "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
        },
        {
            "from": "human",
            "value": textwrap.dedent(f"""\
                I will provide you with 2 passages, each indicated by number identifier [].
                Rank the passages based on their relevance to the search query: {q2}.
            """).strip()
        },
        {
            "from": "gpt",
            "value": "Okay, please provide the passages."
        },
        {
            "from": "human",
            "value": textwrap.dedent(f"""\
                [1] {doc1}
                [2] {doc2}

                Search Query: {q2}.
                Rank the passages above based on their relevance to the search query.
                The passages should be listed in descending order using identifiers.
                The output format should be [] > [], e.g., [1] > [2] or [2] > [1].
                Only respond with the ranking results, do not say any word or explain.
            """).strip()
        },
        {
            "from": "gpt",
            "value": response
        },
    ]
    return conversation

for split_name in splits:
    dataset = load_dataset(dataset_name, split=split_name)
    converted_data = []

    # Step 2: Convert the dataset to the `conversations` key format

    for idx, row in enumerate(dataset):
        if idx % 2 == 1:
            # Create entry with default ranking
            conversation = create_entry(row["q2"], row["doc1"], row["doc2"], swap=False)
        else:
            # Create entry with swapped ranking
            conversation = create_entry(row["q2"], row["doc1"], row["doc2"], swap=True)
        
        converted_data.append({"conversations": conversation})

    # Save converted data for this split
    output_file = f"converted_{split_name}.json"
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=4)

    print(f"Converted {split_name} dataset saved to {output_file}")

    # Add to converted splits
    converted_splits[split_name] = Dataset.from_json(output_file)

# Step 3: Combine splits into a DatasetDict
final_dataset = DatasetDict(converted_splits)

# Step 4: Initialize Hugging Face API and create a new dataset repository
api = HfApi()
# dataset_repo_name = "NevIR-conversational"  # New repository name
dataset_repo_name = "ExcluIR-conversational"  # New repository name

# Create the new dataset repository
# api.create_repo(repo_id=f"thijmennijdam/{dataset_repo_name}", repo_type="dataset", private=False)
# print(f"Dataset repository created: thijmennijdam/{dataset_repo_name}")

# Step 5: Push the combined dataset to Hugging Face
final_dataset.push_to_hub(f"thijmennijdam/{dataset_repo_name}")
print(f"Dataset successfully uploaded to: thijmennijdam/{dataset_repo_name}")

# Cleanup local files
for split_name in splits:
    os.remove(f"converted_{split_name}.json")
print("Temporary files cleaned up.")
