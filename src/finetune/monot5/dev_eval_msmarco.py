import argparse
import pandas as pd
from tqdm import tqdm
from external.pygaggle.rerank.transformer import MonoT5
from external.pygaggle.rerank.base import Query, Text
import jsonlines
import os
from collections import defaultdict
import subprocess
import csv

def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    if '.json' in path:
        with jsonlines.open(path) as reader:
            for obj in tqdm(reader):
                doc_id = str(obj['id'])
                corpus[doc_id] = obj['contents']
    else:  # Assume it's a .tsv
        corpus = pd.read_csv(path, sep='\t', header=None, index_col=0)[1].to_dict()
        corpus = {str(k):v for k,v in corpus.items()}
    return corpus

def load_run(path):
    print('Loading run...')
    run = pd.read_csv(path, delim_whitespace=True, header=None)
    run = run.groupby(0)[1].apply(list).to_dict()
    run = {str(k): [str(d) for d in v] for k,v in run.items()}
    return run

def load_queries(path):
    print('Loading queries...')
    queries = pd.read_csv(path, sep='\t', header=None, index_col=0)
    queries = queries[1].to_dict()
    queries = {str(k):v for k,v in queries.items()}
    return queries

def run_pyserini_eval(qrels_path, marco_path):
    # Run pyserini msmarco passage evaluation and capture output
    cmd = [
        "python", 
        "-m", "pyserini.eval.msmarco_passage_eval", 
        qrels_path,
        marco_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running pyserini evaluation:", result.stderr)
        return None
    return result.stdout

def ensure_csv_exists(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "MRR@10"])  # Example header
    return csv_path

def epoch_already_evaluated(csv_path, epoch):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    return epoch in df['epoch'].tolist()

def parse_mrr_from_pyserini_output(output):
    # pyserini output usually looks like: "MRR @10: 0.1871"
    # We'll try to parse this line.
    for line in output.split('\n'):
        if "MRR @10:" in line:
            parts = line.strip().split()
            # The MRR should be the last token on that line
            # Example line: "MRR @10: 0.1871"
            return float(parts[-1])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="models/checkpoints/monot5/finetune_nevir",
        type=str,
        required=False,
        help="Base path where checkpoints are stored (checkpoint-1, checkpoint-2, etc.)."
    )
    parser.add_argument(
        "--initial_run",
        default="data/MSmarco_data/500top1000.txt",
        type=str,
        required=False,
        help="Path to the initial run file in TREC format."
    )
    parser.add_argument(
        "--corpus",
        default="data/MSmarco_data/collection.tsv",
        type=str,
        required=False,
        help="Path to the document collection."
    )
    parser.add_argument(
        "--queries",
        default="data/MSmarco_data/queries.dev.small.tsv",
        type=str,
        required=False,
        help="Path to the queries file."
    )
    parser.add_argument(
        "--qrels",
        default="data/MSmarco_data/qrels.dev.small.tsv",
        type=str,
        required=False,
        help="Path to the qrels file."
    )
    parser.add_argument(
        "--output_dir",
        default="results/monot5",
        type=str,
        required=False,
        help="Directory to save output runs and CSV results."
    )
    parser.add_argument(
        "--max_epoch",
        default=20,
        type=int,
        required=False,
        help="Maximum epoch (checkpoint) number to evaluate."
    )
    args = parser.parse_args()
    
    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV to store MRR results
    csv_path = os.path.join(args.output_dir, "msmarco_evaluation_results.csv")
    ensure_csv_exists(csv_path)

    print("Loading necessary data...")
    run = load_run(args.initial_run)
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    # Evaluate each checkpoint in a loop
    for epoch in range(1, args.max_epoch+1):
        checkpoint_path = os.path.join(args.model_path, f"checkpoint-{epoch}")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist, skipping...")
            continue

        # Check if this epoch was already evaluated
        if epoch_already_evaluated(csv_path, epoch):
            print(f"Epoch {epoch} already evaluated. Skipping...")
            continue

        print(f"Evaluating epoch {epoch} with checkpoint: {checkpoint_path}")
        #TODO: move model to device
        model = MonoT5(checkpoint_path)

        # Set output file paths for this epoch
        trec_path = os.path.join(args.output_dir, f"outputs/eval_msmarco-epoch{epoch}-trec.txt")
        marco_path = os.path.join(args.output_dir, f"outputs/eval_msmarco-epoch{epoch}-marco.txt")
        if not os.path.exists(os.path.join(args.output_dir, "outputs")):
            os.makedirs(os.path.join(args.output_dir, "outputs"), exist_ok=True)
            
        # Reranking step
        print("Starting reranking...")
        with open(trec_path, 'w') as trec, open(marco_path, 'w') as marco:
            for query_id in tqdm(run.keys()):
                # Rerank for this query
                query = Query(queries[query_id])
                texts = [Text(corpus[doc_id], {'docid': doc_id}, 0) for doc_id in run[query_id]]
                reranked = model.rerank(query, texts)
                for rank, document in enumerate(reranked):
                    trec.write(f'{query_id}\tQ0\t{document.metadata["docid"]}\t{rank+1}\t{document.score}\t{checkpoint_path}\n')
                    marco.write(f'{query_id}\t{document.metadata["docid"]}\t{rank+1}\n')
        print("Reranking done!")

        # Now run pyserini eval
        print("Running pyserini evaluation...")
        pyserini_output = run_pyserini_eval(args.qrels, marco_path)

        if pyserini_output is None:
            print(f"Pyserini evaluation failed for epoch {epoch}, skipping CSV logging.")
            continue

        # Parse MRR from pyserini output
        mrr_value = parse_mrr_from_pyserini_output(pyserini_output)
        if mrr_value is None:
            print("Could not parse MRR@10 from pyserini output.")
            continue

        # Append MRR result to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mrr_value])

        print(f"Epoch {epoch} evaluation complete: MRR@10={mrr_value}")

    print("All evaluations completed.")

if __name__ == "__main__":
    main()
