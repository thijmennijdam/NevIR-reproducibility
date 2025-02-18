import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def transform_to_contrastive_format(corpus_path, queries_path, output_path):
    """
    Loads two JSON files:
      - corpus: A list of strings.
      - queries: A list of dicts with keys "question0", "RQ_rewrite", and "corpus_sub_index".
    Writes a new JSON file with records of the form:
      { "q1": ..., "q2": ..., "doc1": ..., "doc2": ... }
    Returns the list of these records.
    """
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Load queries
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Prepare output
    result = []
    for q in queries:
        q1 = q["question0"]
        q2 = q["RQ_rewrite"]
        doc1_index, doc2_index = q["corpus_sub_index"]

        doc1 = corpus[doc1_index]
        doc2 = corpus[doc2_index]

        entry = {
            "q1": q1,
            "q2": q2,
            "doc1": doc1,
            "doc2": doc2
        }
        result.append(entry)
    
    # Optionally save the transformed data to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":

    # Step 1: Transform to contrastive format
    result = transform_to_contrastive_format(
        corpus_path="data/excluIR/corpus.json",
        queries_path="data/excluIR/test_manual_final.json",
        output_path="contrastive_dataset.json"
    )

    # Step 2: Convert list of dicts to a Dataset
    full_dataset = Dataset.from_list(result)

    # Step 3: Shuffle and split (60% train, 10% val, 30% test)
    #  - First split off 30% test
    #  - Then from the remaining 70%, split off ~14.2857% for val 
    #    (which is 10% of the original total)
    # This yields 60/10/30 overall.

    full_dataset = full_dataset.shuffle(seed=42)
    temp_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)
    train_70 = temp_dataset["train"]  # 70% of data
    test_30 = temp_dataset["test"]    # 30% of data

    # Now split train_70 into train (60% of total) and val (10% of total).
    # That means val should be 10% / 70% = 0.142857 (about 14.2857%)
    splits = train_70.train_test_split(test_size=0.142857, seed=42)
    train_60 = splits["train"]
    val_10 = splits["test"]

    final_dataset = DatasetDict({
        "train": train_60,
        "validation": val_10,
        "test": test_30
    })

    # Step 4: Push dataset to Hugging Face Hub
    api = HfApi()
    dataset_repo_name = "ExcluIR"  # New repository name
    user_id = "thijmennijdam"      # Your HF username

    # Create the new dataset repository (if not already exists)
    api.create_repo(repo_id=f"{user_id}/{dataset_repo_name}", repo_type="dataset", private=False)
    print(f"Dataset repository created (or found): {user_id}/{dataset_repo_name}")

    # Push the dataset
    final_dataset.push_to_hub(f"{user_id}/{dataset_repo_name}")
    print(f"Dataset successfully uploaded to: {user_id}/{dataset_repo_name}")
