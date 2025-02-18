import pandas as pd
from datasets import load_dataset
import os
import logging
import random
import tarfile
import pandas as pd
import polars as pl
import torch
from sentence_transformers import (
    InputExample,
    util,
)
from sentence_transformers.util import http_get
from datasets import load_dataset
import argparse
import gdown
import json

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

def save_samples_to_tsv(samples: list, output_path: str):
    """
    Save a list of InputExample objects to a TSV file.

    Args:
        samples (list): List of InputExample objects.
        output_path (str): Path to save the TSV file.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    data = []
    for example in samples:
        if len(example.texts) == 3:
            query, positive, negative = example.texts
            data.append([query, positive, negative])
        else:
            logging.warning(f"Skipping sample with unexpected format: {example.texts}")

    df = pd.DataFrame(data, columns=["query", "positive", "negative"])
    df.to_csv(output_path, sep="\t", index=False)
    logging.info(f"Combined samples saved to {output_path}")

def download_and_extract_msmarco(data_folder: str):
    """
    Download and extract the MS MARCO dataset if not already present.

    Args:
        data_folder (str): Path to the folder containing the data.
    """
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    dev_queries_file = os.path.join(data_folder, "queries.dev.small.tsv")
    qrels_filepath = os.path.join(data_folder, "qrels.dev.tsv")
    top1000_filepath = os.path.join(data_folder, "top1000.dev")
    
    # Download the collection and queries if not present
    if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
        tar_filepath = os.path.join(data_folder, "collectionandqueries.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info(f"Downloading: {tar_filepath}")
            util.http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
                tar_filepath,
            )
        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    # Download qrels if not present
    if not os.path.exists(qrels_filepath):
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
            qrels_filepath,
        )

    # Download top1000.dev if not present
    if not os.path.exists(top1000_filepath):
        tar_filepath = os.path.join(data_folder, "top1000.dev.tar.gz")
        logging.info("Download: " + tar_filepath)
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz", tar_filepath
        )
        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


def load_msmarco_train_samples(data_folder: str, times_nevir_size: int = 1):
    """
    Loads a random subset of MS MARCO training samples using Polars for faster CSV reading.

    Args:
        data_folder (str): Path to the folder containing the data.
        times_nevir_size (int): Factor to multiply the MSMarco training data size 
        by the size of NevIR data.

    Returns:
        list: A list of InputExample objects.
    """
    triplet_filepath = os.path.join(data_folder, "triples.train.small.tsv")

    # Check if the file exists, otherwise download and extract it
    if not os.path.exists(triplet_filepath):
        tar_filepath = os.path.join(data_folder, "triples.train.small.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Downloading: " + tar_filepath)
            http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
                tar_filepath
            )
        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    # Load the data using Polars and perform random sampling
    number_of_train_samples = 1896 * times_nevir_size
    df = pl.read_csv(triplet_filepath, separator="\t", has_header=False, new_columns=["query", "positive", "negative"])
    sampled_df = df.sample(n=number_of_train_samples, seed=42)

    # Convert to InputExample format
    samples = [
        InputExample(texts=[row["query"], row["positive"], row["negative"]])
        for row in sampled_df.to_dicts()
    ]

    return samples

def convert_nevir_dataset_to_triplets(df: pd.DataFrame, evaluation: bool = False) -> list:
    """
    Convert a DataFrame to a list of triplets for training or evaluation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with queries and documents.
    - evaluation (bool): If True, generate evaluation triplets; otherwise, generate training triplets.

    Returns:
    - list: A list of triplet dictionaries.
    """
    all_instances = []

    for _, row in df.iterrows():
        if not evaluation:
            all_instances.extend([
                {"query": row["q1"], "positive": row["doc1"], "negative": row["doc2"]},
                {"query": row["q2"], "positive": row["doc2"], "negative": row["doc1"]},
            ])
        else:
            all_instances.append({
                "query1": row["q1"], "query2": row["q2"],
                "positive": row["doc1"], "negative": row["doc2"]
            })

    print(f"Number of instances: {len(all_instances)}")
    return all_instances

def process_and_save_nevir_split(dataset_name: str, split: str, evaluation: bool, output_file: str):
    """
    Load a dataset split, convert it to triplets, and save to a TSV file.

    Parameters:
    - dataset_name (str): The Hugging Face dataset name.
    - split (str): The dataset split ('train', 'test', or 'validation').
    - evaluation (bool): If True, generate evaluation triplets; otherwise, generate training triplets.
    - output_file (str): The output file path for saving triplets.
    """
    # Load the dataset split
    try:
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()
    except Exception as e:
        print(f"Error loading dataset split '{split}': {e}")
        return

    # Generate triplet examples
    triplet_examples = convert_nevir_dataset_to_triplets(df, evaluation)

    # Convert to DataFrame and save to TSV
    triplets_df = pd.DataFrame(triplet_examples)
    triplets_df.to_csv(output_file, sep="\t", index=False)
    print(f"Triplets saved to '{output_file}'")

def prepare_nevir(nevir_data_path: str):
    """
    Prepare the NevIR dataset by processing the train, test, and validation splits.

    Args:
        nevir_data_path (str): Path to the NevIR dataset.
    """
    dataset_name = "orionweller/NevIR"

    if not os.path.exists(nevir_data_path):
        os.makedirs(nevir_data_path)
        
    # Process train split
    process_and_save_nevir_split(
        dataset_name=dataset_name,
        split="train",
        evaluation=False,
        output_file=nevir_data_path + "train_triplets.tsv"
    )

    # Process test split
    process_and_save_nevir_split(
        dataset_name=dataset_name,
        split="test",
        evaluation=True,
        output_file=nevir_data_path + "test_triplets.tsv"
    )

    # Process validation split
    process_and_save_nevir_split(
        dataset_name=dataset_name,
        split="validation",
        evaluation=True,
        output_file=nevir_data_path + "validation_triplets.tsv"
    )

def load_nevir_data(df: pd.DataFrame, train: bool = True):
    """
    Load the NevIR dataset from a DataFrame and convert it to InputExample objects.

    Args:
        df (pd.DataFrame): The input DataFrame.
        train (bool): If True, load training samples; otherwise, load evaluation samples.

    Returns:
        list: A list of InputExample objects.
    """
    samples = []
    if train:
        for i, row in df.iterrows():
            samples.append(InputExample(texts=[row['query'], row['positive'], row['negative']]))
    else:   
        for i, row in df.iterrows():
            samples.append(InputExample(texts=[row["query1"], row["query2"], row["positive"], row["negative"]]))

    return samples

def create_MSMarco_dev_subset(rank_file: str, output_path: str, dev_samples: int = 500):
    """
    Reads the rank file, selects a random subset of qids, and writes the filtered
    data to the output path.
    
    Args:
        rank_file (str): Path to the input rank file.
        output_path (str): Path to the output file.
        dev_samples (int): Number of random dev samples to load.
    """
    # Read the input file
    msmarco_dev_rank = pd.read_csv(rank_file, sep="\t", header=None, names=["qid", "pid", "query", "passage"])
    
    # Get unique qids and shuffle them
    unique_elements = msmarco_dev_rank['qid'].unique()
    random.shuffle(unique_elements)
    
    # Select random qids
    random_qids = unique_elements[:dev_samples]
    
    # Create a new dataframe for the filtered results
    top1000_df = pd.DataFrame(columns=['qid', 'pid', 'query', 'passage'])
    
    for r_qid in random_qids:
        top1000_documents = msmarco_dev_rank[msmarco_dev_rank['qid'] == r_qid]
        top1000_df = pd.concat([top1000_df, top1000_documents], ignore_index=True)
    
    # Write the resulting dataframe to the output file
    top1000_df.to_csv(output_path, sep='\t', index=False, header=False)


    # Select random qids
    random_qids = unique_elements[dev_samples:2*dev_samples]
    
    # Create a new dataframe for the filtered results
    top1000_df = pd.DataFrame(columns=['qid', 'pid', 'query', 'passage'])
    
    for r_qid in random_qids:
        top1000_documents = msmarco_dev_rank[msmarco_dev_rank['qid'] == r_qid]
        top1000_df = pd.concat([top1000_df, top1000_documents], ignore_index=True)
    
    # Write the resulting dataframe to the output file
    top1000_df.to_csv(f'{output_path}_test', sep='\t', index=False, header=False)



def prepare_ExcluIR_NevIR(corpus_file: str, test_manual_final_file: str, output_path_excluIR: str, nevir_file: str, output_path_combi: str, output_path_excluIR_test: str, output_path_excluIR_val: str,):
    """
    Prepare the ExcluIR and NevIR datasets for training by creating triplets.

    Args:
        corpus_file (str): Path to the ExcluIR corpus JSON file.
        test_manual_final_file (str): Path to the ExcluIR test manual final JSON file.
        output_path_excluIR (str): Path to save the ExcluIR TSV output.
        nevir_file (str): Path to the NevIR TSV file.
        output_path_combi (str): Path to save the combined ExcluIR and NevIR TSV output.
    """
    # Load JSON data from a file
    with open(test_manual_final_file, 'r') as file:
        data = json.load(file)

    if not os.path.exists(os.path.dirname(output_path_excluIR)):
        os.makedirs(os.path.dirname(output_path_excluIR))
    
    if not os.path.exists(os.path.dirname(output_path_combi)):
        os.makedirs(os.path.dirname(output_path_combi))

    if not os.path.exists(os.path.dirname(output_path_excluIR_test)):
        os.makedirs(os.path.dirname(output_path_excluIR_test))

    if not os.path.exists(os.path.dirname(output_path_excluIR_val)):
        os.makedirs(os.path.dirname(output_path_excluIR_val))
         
    # Normalize JSON data and create a DataFrame
    df = pd.json_normalize(data)

    # Extract `positive` and `negative` from `corpus_sub_index`
    df['doc1_id'] = df['corpus_sub_index'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['doc2_id'] = df['corpus_sub_index'].apply(lambda x: x[1] if len(x) > 1 else None)

    # Map fields to desired column names
    df = df.rename(columns={
        "question0": "query1",
        "RQ_rewrite": "query2"
    })

    # Remove \n and \r from both query columns
    df['query1'] = df['query1'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' ') if x is not None else x)
    df['query2'] = df['query2'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' ') if x is not None else x)

    # Drop the original `corpus_sub_index` column as it's no longer needed
    df_main = df.drop(columns=['corpus_sub_index'])

    # Load JSON data from a file
    with open(corpus_file, 'r') as file:
        data = json.load(file)

    # Create a DataFrame with doc_id and document columns
    df_corpus = pd.DataFrame({
        'doc_id': range(len(data)),  # Assign incremental IDs
        'document': data            # Assign the corresponding documents
    })

    # Clean the document text by replacing newline characters with spaces
    df_corpus["document"] = df_corpus["document"].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' '))

    # Map positive and negative IDs to their documents
    df_main["doc1"] = df_main["doc1_id"].map(
        df_corpus.set_index("doc_id")["document"]
    )
    df_main["doc2"] = df_main["doc2_id"].map(
        df_corpus.set_index("doc_id")["document"]
    )

    rows = []

    # Loop through each entry and add rows systematically
    for idx, row in df_main.iterrows():
        # Add the row corresponding to query1 as the first triplet
        rows.append({
            "query": row["query1"],
            "positive": row["doc1"],
            "negative": row["doc2"]
        })
        # Add the row corresponding to query2 as the second triplet
        rows.append({
            "query": row["query2"],
            "positive": row["doc2"],
            "negative": row["doc1"]
        })

    # Convert the list of rows into a new DataFrame
    df_triplets = pd.DataFrame(rows)

    # Get the first 1892 elements
    df_triplets_train = df_triplets[:1892]

    # Save the DataFrame to a TSV file ensuring that each row is on one line and columns are tab-separated
    df_triplets_train.to_csv(output_path_excluIR, sep="\t", index=False, header=False)

    # Define the column names
    column_names = ['query', 'positive', 'negative']

    # Read the TSV files into DataFrames (assuming no headers in the files)
    df_excluir = pd.read_csv(output_path_excluIR, sep='\t', header=None, names=column_names)
    df_nevir = pd.read_csv(nevir_file, sep='\t', header=None, names=column_names)

    # Concatenate the two DataFrames
    df_combined = pd.concat([df_excluir, df_nevir], ignore_index=True)

    # Save the combined DataFrame as a new TSV file
    df_combined.to_csv(output_path_combi, sep='\t', index=False)
    print(f'saved combi file to {output_path_combi}')

    # save test set
    subset_test = df_main.iloc[946:1383+946]
    subset = subset_test[['query1', 'query2', 'doc1', 'doc2']]
    subset.to_csv(output_path_excluIR_test, sep="\t", index=False, header=False)

    # save validation set
    subset_val = df_main.iloc[1383+946:1383+946+225]
    subset = subset_val[['query1', 'query2', 'doc1', 'doc2']]
    subset.to_csv(output_path_excluIR_val, sep="\t", index=False, header=False)

    print(f'saved exluIR test set to {output_path_excluIR_test}')


def download_excluIR(test_manual_final_file: str, corpus_file: str):
    """
    Download the ExcluIR dataset if not already present.

    Args:
        test_manual_final_file (str): Path to the ExcluIR test manual final JSON file.
        corpus_file (str): Path to the ExcluIR corpus JSON file.
    """
    if not os.path.exists(test_manual_final_file) or not os.path.exists(corpus_file):
        if not os.path.exists(os.path.dirname(test_manual_final_file)):
            os.makedirs(os.path.dirname(test_manual_final_file))
        if not os.path.exists(os.path.dirname(corpus_file)):
            os.makedirs(os.path.dirname(corpus_file))
            
    test_manual_url ='https://drive.google.com/uc?id=1PwDeIPdGu4T2uCdhvzdVepdqdV6CzzgL'
    corpus_url = 'https://drive.google.com/uc?id=18-ODtPKGH3KC3_KijoobPHbxzDHWeUYv'
    gdown.download(test_manual_url, test_manual_final_file, quiet=False)
    gdown.download(corpus_url, corpus_file, quiet=False)
    print(f'succesfully downloaded excluIR')

def convert_dev_to_trec(input_file, output_file):
    """
    Convert the first `max_lines` of a .dev file to a TREC-style .txt file.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        i = 0 
        for idx, line in enumerate(infile):
            parts = line.strip().split("\t")  # Adjust separator if needed
            if len(parts) >= 2:  # Ensure at least query_id and doc_id are present
                query_id = parts[0]
                doc_id = parts[1]
                # Write TREC-style output with placeholders for rank, score, and method
                outfile.write(f"{query_id} {doc_id} {idx + 1} 1.0 BM25\n")
            else:
                print(f"Skipping malformed line: {line.strip()}")

    print(f"Conversion complete. Output saved to: {output_file}")
    
def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Prepare datasets for training.")
    parser.add_argument('--data-dir', required=False, default="data/", help="Base directory for storing all datasets.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--dev_samples', type=int, default=500, help="Number of random dev samples to load.")

    args = parser.parse_args()
    data_dir = args.data_dir

    # Define all data paths relative to --data-dir
    nevir_data_path = os.path.join(data_dir, "NevIR_data/")
    msmarco_data_path = os.path.join(data_dir, "MSmarco_data/")
    merged_dataset_path = os.path.join(data_dir, "merged_dataset/")
    rank_file = os.path.join(data_dir, "MSmarco_data/top1000.dev")
    output_path = os.path.join(data_dir, "MSmarco_data/500top1000.dev")
    corpus_file = os.path.join(data_dir, "ExcluIR_data/corpus.json")
    test_manual_file = os.path.join(data_dir, "ExcluIR_data/test_manual_final.json")
    output_excluIR = os.path.join(data_dir, "ExcluIR_data/train_triplets.tsv")
    nevir_file = os.path.join(data_dir, "NevIR_data/train_triplets.tsv")
    output_combi = os.path.join(data_dir, "Exclu_NevIR_data/combined_train_triplets.tsv")
    output_path_excluIR_test = os.path.join(data_dir, "ExcluIR_data/excluIR_test.tsv")
    output_path_excluIR_val = os.path.join(data_dir, "ExcluIR_data/excluIR_val.tsv")

    _set_seed(args.seed)

    print("Preparing NevIR data...")
    prepare_nevir(nevir_data_path)

    print("Downloading MS MARCO data if needed...")
    download_and_extract_msmarco(msmarco_data_path)

    print("Loading MS MARCO training data...")
    train_samples_msmarco = load_msmarco_train_samples(msmarco_data_path)

    print("Loading NevIR data...")
    train_df = pd.read_csv(nevir_file, sep="\t")
    train_samples_nevir = load_nevir_data(train_df, train=True)

    print("Saving combined dataset...")
    all_samples = train_samples_nevir + train_samples_msmarco
    save_samples_to_tsv(all_samples, os.path.join(merged_dataset_path, "train_samples.tsv"))

    print("Creating MS MARCO dev subset...")
    create_MSMarco_dev_subset(rank_file, output_path, args.dev_samples)

    print("Downloading ExcluIR...")
    download_excluIR(test_manual_file, corpus_file)

    print("Preparing ExcluIR and NevIR datasets...")
    prepare_ExcluIR_NevIR(
        corpus_file=corpus_file,
        test_manual_final_file=test_manual_file,
        output_path_excluIR=output_excluIR,
        nevir_file=nevir_file,
        output_path_combi=output_combi,
        output_path_excluIR_test= output_path_excluIR_test,
        output_path_excluIR_val=output_path_excluIR_val
    )

    print("Converting dev file to TREC format in txt file (needed for eval MSMarco on monot5)...")
    convert_dev_to_trec(input_file=output_path, output_file=output_path.replace('.dev', '.txt'))
    
    print("Converting dev file to TREC format in txt file (needed for eval MSMarco on monot5)...")
    new_output_path = os.path.join(data_dir, "MSmarco_data/500top1000_test.txt")
    convert_dev_to_trec(input_file=output_path+'_test', output_file=new_output_path)
    

if __name__ == "__main__":
    main()

# example usage
# python prepare_ExcluIR_NevIR.py \
# --corpus_file data/ExcluIR_data/corpus.json \
# --test_manual_final_file data/ExcluIR_data/test_manual_final.json \
# --output_path_excluIR data/ExcluIR_data/train_triplets.tsv \
# --nevir_file data/NevIR_data/train_triplets.tsv \
# --output_path_combi data/Exclu_NevIR_data/combined_train_triplets.tsv \
# --rank_file data/MSmarco_data/top1000.dev \
# --output_path data/MSmarco_data/500top1000.dev \
# --seed 42 \
# --dev_samples 500 \
# --nevir_data_path data/NevIR_data/ \
# --msmarco_data_path data/MSmarco_data/ \
# --merged_dataset_path data/merged_dataset/ \

