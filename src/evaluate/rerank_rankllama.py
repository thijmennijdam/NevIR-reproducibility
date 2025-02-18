import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import PeftModel, PeftConfig
import numpy as np
import pandas as pd
import argparse

set_seed(1)

def load_rankllama_model(peft_model_name="castorini/rankllama-v1-7b-lora-doc"):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=1
    )
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    model.to("cuda")
    return model

def calc_preferred_rankllama(
    doc1, doc2, q1, q2, model_name="castorini/rankllama-v1-7b-lora-doc", model=None
):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q1, q2: strings for queries that are relevant to the corresponding docs
        model_name: string containing the RankLLaMA checkpoint name
        model: if provided, a pre-loaded (model, tokenizer) tuple for reuse

    Returns:
        (results, model, similarity_score)

        results: dict containing:
            "q1": [score_for_doc1, score_for_doc2]
            "q2": [score_for_doc1, score_for_doc2]
            "score": a float between 0 and 1 representing how many queries ranked their intended doc higher.

        model: If a model was loaded here, returns (rankllama_model, tokenizer) tuple for caching.
        similarity_score: None, included for interface compatibility.
    """
    if model is None:
        # If no model is provided, load it
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        rankllama_model = load_rankllama_model(model_name)
        model = (rankllama_model, tokenizer)
    else:
        rankllama_model, tokenizer = model

    # Assign docids
    passages = [(1, doc1), (2, doc2)]
    queries = [q1, q2]
    results = {}
    num_correct = 0
    similarity_score = None

    
    # only do first 10 rows
    i = 0
    for idx, query in enumerate(queries):
        i += 1
        if i > 10:
            break   
        # Score each document for the given query
        doc_scores = []
        for (docid, doc) in passages:
            inputs = tokenizer(
                f"query: {query}",
                f"document: {doc}",
                return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                outputs = rankllama_model(**inputs)
                logits = outputs.logits
                score = float(logits[0][0].item())
            doc_scores.append((docid, score))

        # Extract scores for doc1 and doc2
        doc1_score = next(s for (d, s) in doc_scores if d == 1)
        doc2_score = next(s for (d, s) in doc_scores if d == 2)
        scores = [doc1_score, doc2_score]

        results[f"q{idx+1}"] = scores
        # The "correct" doc is the one that corresponds to this query:
        # For q1, doc1 should have the higher score; for q2, doc2 should have the higher score.
        should_be_higher = scores[idx]
        should_be_lower = scores[0] if idx != 0 else scores[1]

        if should_be_higher > should_be_lower:
            num_correct += 1

    results["score"] = num_correct / 2.0
    return results, model, similarity_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="whether to load a file and if so, the path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d1",
        "--doc1",
        help="doc1 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d2",
        "--doc2",
        help="doc2 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-q1",
        "--q1",
        help="q1 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-q2",
        "--q2",
        help="q2 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="the model to use, if not loading from file",
        type=str,
        default="castorini/rankllama-v1-7b-lora-doc",
    )
    args = parser.parse_args()

    if not args.file and (
        not args.doc1 or not args.doc2 or not args.q1 or not args.q2
    ):
        print(
            "Error: need either a file path or the input args (d1, d2, q1, q2, model)"
        )
    elif args.file:
        print("Loading from file...")
        df = pd.read_csv(args.file)
        for (idx, row) in df.iterrows():
            print(
                calc_preferred_rankllama(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    row.get("model_name", args.model_name),
                )[0]
            )
    else:
        print("Loading from args...")
        print(
            calc_preferred_rankllama(
                args.doc1, args.doc2, args.q1, args.q2, args.model_name
            )[0]
        )
