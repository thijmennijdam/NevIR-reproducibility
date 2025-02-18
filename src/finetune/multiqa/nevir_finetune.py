import os
import logging
import random

import pandas as pd
from torch.utils.data import DataLoader
import torch
import argparse

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    SentencesDataset,
    evaluation
)
from finetune.multiqa.pairwise_evaluator import PairwiseEvaluator

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

#TODO: move to utils
def load_nevir_data(df, train=False):
    """
    Load the NevIR dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the NevIR data.
        train (bool): Whether to load the training or validation data.

    Returns:
        List[InputExample]: List of InputExample objects containing the data.
    """
    samples = []
    if train:
        for i, row in df.iterrows():
            samples.append(InputExample(texts=[row['query'], row['positive'], row['negative']]))
    else:   
        for i, row in df.iterrows():
            samples.append(InputExample(texts=[row["query1"], row["query2"], row["positive"], row["negative"]]))
    
    # only do first 10 instances
    return samples

#TODO: move to utils
def load_msmarco_topk1000_dev(data_folder: str):
    """
    Load the MS MARCO dev dataset with the top 1000 passages per query.

    Args:
        data_folder (str): Path to the MS MARCO data folder.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]: Tuple containing the dev queries, dev corpus, and relevant documents.
    """
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    dev_queries_file = os.path.join(data_folder, "queries.dev.small.tsv")
    qrels_filepath = os.path.join(data_folder, "qrels.dev.tsv")
    top1000_rank_filepath = os.path.join(data_folder, "500top1000.dev")


    # Initialize dictionaries
    dev_queries_full = {}
    corpus = {}
    dev_rel_docs_full = {}
    top1000_pids = set()

    # Load the dev queries
    with open(dev_queries_file, encoding="utf8") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            dev_queries_full[qid] = query.strip()

    # Load the relevant passages for each query (qrels)
    with open(qrels_filepath) as f:
        for line in f:
            qid, _, pid, _ = line.strip().split("\t")
            if qid not in dev_queries_full:
                continue
            if qid not in dev_rel_docs_full:
                dev_rel_docs_full[qid] = set()
            dev_rel_docs_full[qid].add(pid)

    # Load the top 1000 ranked passages per query
    top1000_rank = {}
    with open(top1000_rank_filepath) as f:
        for line in f:
            qid, pid, _, _ = line.strip().split("\t")
            if qid not in dev_queries_full:
                continue
            if qid not in top1000_rank:
                top1000_rank[qid] = []
            if len(top1000_rank[qid]) < 1000:
                top1000_rank[qid].append(pid)
                top1000_pids.add(pid)

    # Load the passages from the collection (only top 1000 pids)
    with open(collection_filepath, encoding="utf8") as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            if pid in top1000_pids:
                corpus[pid] = passage.strip()

    dev_query_ids = list(top1000_rank.keys())

    # Filter the dev data to only include the top 1000 ranked passages
    dev_queries = {qid: dev_queries_full[qid] for qid in dev_query_ids if qid in dev_queries_full}
    dev_rel_docs = {qid: dev_rel_docs_full[qid] for qid in dev_query_ids if qid in dev_rel_docs_full}
    dev_pids = {pid for qid in dev_query_ids for pid in top1000_rank.get(qid, [])}
    dev_corpus = {pid: corpus[pid] for pid in dev_pids if pid in corpus}

    return dev_queries, dev_corpus, dev_rel_docs


def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/multi-qa-mpnet-base-dot-v1", help="Model name to use.")
    parser.add_argument("--msmarco_path", type=str, default="data/MSmarco_data/", help="Path to MS MARCO data.")
    parser.add_argument("--nevir_path", type=str, default="data/NevIR_data/", help="Path to NevIR data.")
    parser.add_argument("--output_path", type=str, default="multiqa/finetune_NevIR/", help="Path to save the model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training.")

    args = parser.parse_args()

    _set_seed(args.seed)

    model = SentenceTransformer(args.model_name)

    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Using GPU")
    else:
        print("Using CPU")

    print("Loading MS MARCO dev data...")
    dev_queries, dev_corpus, dev_rel_docs = load_msmarco_topk1000_dev(
        args.msmarco_path)

    print("Loading NevIR train and dev data...")
    train_df = pd.read_csv(args.nevir_path + "train_triplets.tsv", sep="\t")
    validation_df = pd.read_csv(args.nevir_path + "validation_triplets.tsv", sep="\t")
    train_samples_nevir = load_nevir_data(train_df, train=True)
    validation_samples_nevir = load_nevir_data(validation_df, train=False)

    train_dataset = SentencesDataset(train_samples_nevir, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    print("Preparing evaluators...")
    evaluator_val_msmarco = evaluation.InformationRetrievalEvaluator(
        queries=dev_queries,
        corpus=dev_corpus,
        relevant_docs=dev_rel_docs,
        name="msmarco-dev"
    )

    evaluator_val_nevir = PairwiseEvaluator.from_input_examples(validation_samples_nevir, name="val", model_name=args.model_name)

    combined_evaluator = evaluation.SequentialEvaluator(
        [evaluator_val_msmarco, evaluator_val_nevir]
    )


    print("Starting training...")
    safe_model_name = args.model_name.replace("/", "_")
    train_loss = losses.TripletLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=combined_evaluator,
        epochs=args.epochs,
        optimizer_params={'lr': args.lr},   
        checkpoint_save_total_limit=2,
        warmup_steps=500,
        save_best_model=True,
        checkpoint_path=f"models/checkpoints/{args.output_path}{safe_model_name}/checkpoints",
        output_path=f"results/{args.output_path}{safe_model_name}/model"
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
