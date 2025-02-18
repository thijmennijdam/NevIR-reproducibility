import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from external.rankgpt.rank_gpt_utils import sliding_windows
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate re-ranking on MSMARCO with multiple checkpoints.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Single model name to use if not looping over checkpoints.")
    parser.add_argument("--checkpoint_base_dir", type=str, default=None,
                        help="Directory containing multiple checkpoints named like 'checkpoint-XXXX'.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Directory to store CSV results and JSON caches.")
    parser.add_argument("--use_cache", action="store_true",
                        help="If True, load results from cache when available; otherwise re-run the model.")
    parser.add_argument("--window_size", type=int, help="Sliding window size.", default=4)
    parser.add_argument("--step", type=int, help="Sliding window step.", default=2)
    
    return parser.parse_args()

def evaluate_mrr_at_k(rankings, ground_truth, k=10, min_rank_threshold=None):
    """
    Calculate MRR@K (Mean Reciprocal Rank) for a given set of rankings and ground truth relevance.
    
    Parameters:
    - rankings: DataFrame containing ranked results with columns ["qid", "pid", "score"].
    - ground_truth: DataFrame containing relevance labels with columns ["qid", "pid", "relevance"].
    - k: The cutoff rank for MRR computation.
    - min_rank_threshold: If set, queries where all relevant documents are ranked below this threshold 
      will still be included, but only valid queries will count in the final MRR denominator.

    Returns:
    - MRR@K score
    """
    if rankings.empty:
        return 0.0

    mrr = 0.0
    total_queries = rankings["qid"].nunique()
    valid_query_count = total_queries  # Default to using all queries
    queries_below_threshold = 0

    for qid, group in rankings.groupby("qid"):
        relevant_pids = set(ground_truth[ground_truth["qid"] == qid]["pid"].tolist())
        ranked_pids = group["pid"].tolist()[:k]  # Only consider top-k

        # If min_rank_threshold is set, count how many queries are valid
        if min_rank_threshold is not None:
            full_ranking = group["pid"].tolist()  # Consider full ranking
            min_relevant_rank = min(
                (full_ranking.index(pid) + 1 for pid in relevant_pids if pid in full_ranking),
                default=float("inf"),
            )
            if min_relevant_rank > min_rank_threshold:
                queries_below_threshold += 1  # Mark this query as below threshold

        # Compute Reciprocal Rank
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid in relevant_pids:
                mrr += 1.0 / rank
                break  # Stop at the first relevant doc

    # Adjust valid query count if threshold is set
    if min_rank_threshold is not None:
        valid_query_count = total_queries - queries_below_threshold

    print(f"Queries with relevant docs ranked below {min_rank_threshold}: {queries_below_threshold}")
    print(f"Valid queries for MRR@{k}: {valid_query_count}")
    # Compute final MRR using valid query count (or total if no threshold)
    return mrr / valid_query_count if valid_query_count > 0 else 0.0



def load_model_and_tokenizer(checkpoint_path):
    """
    Load a FastLanguageModel and tokenizer from a given checkpoint path or Hugging Face model name.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True,
    )
    return model, tokenizer


###############################################################################
# BM25 caching (and computation) 
###############################################################################
def load_or_compute_bm25(msmarco_data, n_ranks_bm25, output_folder, use_cache):
    """
    Compute BM25 rankings (top n_ranks_bm25) for each query OR load cached results.
    Returns:
      - bm25_rankings: DataFrame with columns [qid, pid] in BM25 order.
      - queries_dict: dict mapping each qid to {"query_text": str, "topn_docs": DataFrame}
    """
    cache_path = os.path.join(output_folder, "bm25_cache.json")
    if use_cache and os.path.exists(cache_path):
        print(f"[Cache] Loading BM25 results from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        queries_dict = {}
        rows = []
        for qid_str, info in data.items():
            qid = int(qid_str)
            query_text = info["query_text"]
            # Convert list-of-dicts back to DataFrame
            topn_docs_df = pd.DataFrame(info["topn_docs"])
            queries_dict[qid] = {"query_text": query_text, "topn_docs": topn_docs_df}
            for _, row in topn_docs_df.iterrows():
                rows.append((qid, row["pid"]))
        bm25_rankings = pd.DataFrame(rows, columns=["qid", "pid"])
        return bm25_rankings, queries_dict
    else:
        print("Computing BM25 rankings...")
        bm25_rows = []
        queries_dict = {}
        query_ids = msmarco_data['qid'].unique()
        for qid in tqdm(query_ids, desc="BM25 Ranking"):
            query_data = msmarco_data[msmarco_data['qid'] == qid].copy()
            query_text = query_data['query'].iloc[0]
            bm25 = BM25Okapi(query_data['passage'].str.split().tolist())
            bm25_scores = bm25.get_scores(query_text.split())
            query_data['bm25_score'] = bm25_scores
            topn_docs = query_data.sort_values(by='bm25_score', ascending=False).head(n_ranks_bm25)
            # Save BM25 ranking (we use the whole row for later passage info)
            queries_dict[qid] = {
                'query_text': query_text,
                # Convert to a list of dicts for JSON serialization
                'topn_docs': topn_docs.to_dict(orient='records')
            }
            for _, row in topn_docs.iterrows():
                bm25_rows.append((qid, row["pid"]))
        bm25_rankings = pd.DataFrame(bm25_rows, columns=["qid", "pid"])
        # Save cache (note: topn_docs is a list of dicts, which is JSON-serializable)
        with open(cache_path, "w") as f:
            json.dump({str(qid): info for qid, info in queries_dict.items()}, f, indent=2)
        # Convert the cached topn_docs back to DataFrames in queries_dict
        for qid, info in queries_dict.items():
            info["topn_docs"] = pd.DataFrame(info["topn_docs"])
        return bm25_rankings, queries_dict


###############################################################################
# Existing sliding window re-ranking with caching (unchanged except for a new parameter)
###############################################################################
def rerank_with_cache(
    checkpoint_identifier,
    n_ranks,
    queries_dict,
    output_folder,
    qrels_df,
    use_cache=False,
    model=None,
    tokenizer=None,
    window_size=4,
    step=2
):
    """
    Re-rank queries using sliding_windows and cache the results.
    Uses the (Jina-updated) queries_dict.
    """
    safe_identifier = re.sub(r"[^\w\-]+", "_", checkpoint_identifier)
    cache_path = os.path.join(output_folder, f"outputs_{safe_identifier}.json")

    if use_cache and os.path.exists(cache_path):
        print(f"[Cache] Found cached sliding-window results for '{checkpoint_identifier}'. Loading...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        all_rows = []
        for qid_str, pids_list in data["results"].items():
            qid_val = int(qid_str)
            for pid in pids_list:
                all_rows.append((qid_val, pid))
        return pd.DataFrame(all_rows, columns=["qid", "pid"])

    if model is None or tokenizer is None:
        raise ValueError("Model/Tokenizer must be provided for re-ranking (cache missing or use_cache=False).")

    if use_cache:
        print(f"[Cache] use_cache=True but no cached sliding-window results for '{checkpoint_identifier}', re-ranking now...")
    else:
        print(f"[Cache] use_cache=False => ignoring any cached sliding-window results and re-ranking '{checkpoint_identifier}' now...")

    reranked_dict = {}

    for qid, qinfo in tqdm(queries_dict.items(), desc=f"Sliding-window reranking with {checkpoint_identifier}"):
        query_text = qinfo['query_text']
        topn_docs_df = qinfo['topn_docs']

        # If no relevant doc in the candidate list, skip re-ranking.
        relevant_pids_for_qid = set(qrels_df[qrels_df["qid"] == qid]["pid"])
        candidate_pids = set(topn_docs_df["pid"])
        if len(relevant_pids_for_qid.intersection(candidate_pids)) == 0:
            reranked_dict[qid] = topn_docs_df["pid"].tolist()
            print(f"Query {qid} has no relevant docs in top-{n_ranks} candidates. Skipping re-ranking.")
            continue

        df_input = pd.DataFrame({
            'query': [query_text],
            'hits': [[
                {'pid': pid, 'content': passage}
                for pid, passage in zip(topn_docs_df['pid'], topn_docs_df['passage'])
            ]]
        })
            
        for _, item in df_input.iterrows():
            reranked_item = sliding_windows(
                item,
                rank_start=0,
                rank_end=n_ranks,
                window_size=window_size,
                step=step,
                model_name=checkpoint_identifier,
                model=model,
                tokenizer=tokenizer
            )
            passages = reranked_item["hits"]
            reranked_dict[qid] = [p["pid"] for p in passages]

    data_to_save = {
        "metadata": {"checkpoint_identifier": checkpoint_identifier},
        "results": {str(qid): pids for qid, pids in reranked_dict.items()}
    }
    with open(cache_path, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"[Cache] Saved sliding-window re-ranking results to '{cache_path}'")

    all_rows = []
    for qid, pids_list in reranked_dict.items():
        for pid in pids_list:
            all_rows.append((qid, pid))
    return pd.DataFrame(all_rows, columns=["qid", "pid"])


def save_results_to_csv(results, output_folder, filename="evaluation_results.csv"):
    """
    Saves the results DataFrame incrementally, appending if the file exists.
    """
    csv_path = os.path.join(output_folder, filename)
    results_df = pd.DataFrame(results)
    
    # Append mode: If file exists, don't write headers again
    results_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    
    print(f"Results saved to {csv_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    file_path = "/scratch-shared/tnijdam_ir2_other/MSmarco_data/500top1000.dev"
    qrels_path = "/scratch-shared/tnijdam_ir2_other/MSmarco_data/qrels.dev.tsv"

    n_ranks_bm25 = 30

    msmarco_data = pd.read_csv(file_path, sep="\t", header=None, names=["qid", "pid", "query", "passage"])
    msmarco_data = msmarco_data.groupby('qid').filter(lambda x: len(x) >= n_ranks_bm25)
    
    print("Number of queries:", msmarco_data['qid'].nunique())
    
    # unique_qids = sorted(msmarco_data['qid'].unique())[:50]
    # msmarco_data = msmarco_data[msmarco_data['qid'].isin(unique_qids)]
    
    qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "0", "pid", "relevance"])

    # === Step 1: Compute or Load BM25 Rankings ===
    bm25_rankings, queries_dict = load_or_compute_bm25(msmarco_data, n_ranks_bm25, args.output_folder, args.use_cache)
    bm25_mrr = evaluate_mrr_at_k(bm25_rankings, qrels, k=10)

    print("BM25 MRR@10:", bm25_mrr)

    results = [{
        "epoch": 0,
        "model": "BM25",
        "MRR@10": bm25_mrr
    }]
    
    # Save **BM25** results immediately
    save_results_to_csv(results, args.output_folder)

    # === Step 2: Re-Ranking with Checkpoints ===
    def extract_checkpoint_number(dirname):
        match = re.search(r'checkpoint-(\d+)', dirname)
        return int(match.group(1)) if match else -1

    epoch_counter = 1  # Start at 1 since BM25 is epoch 0

    if args.checkpoint_base_dir:
        dirs = sorted([
            d for d in os.listdir(args.checkpoint_base_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.checkpoint_base_dir, d))
        ], key=extract_checkpoint_number)

        for subdir in dirs:
            full_ckpt_path = os.path.join(args.checkpoint_base_dir, subdir)
            checkpoint_identifier = subdir

            model, tokenizer = None, None
            if not args.use_cache:
                print(f"Use_cache=False => forcing a re-run for {checkpoint_identifier}")
                model, tokenizer = load_model_and_tokenizer(full_ckpt_path)
            else:
                safe_id = re.sub(r"[^\w\-]+", "_", subdir)
                cache_path = os.path.join(args.output_folder, f"outputs_{safe_id}.json")
                if not os.path.exists(cache_path):
                    print(f"Use_cache=True but no existing cache for {checkpoint_identifier}, loading model.")
                    model, tokenizer = load_model_and_tokenizer(full_ckpt_path)

            reranked_df = rerank_with_cache(
                checkpoint_identifier=checkpoint_identifier,
                n_ranks=n_ranks_bm25,
                queries_dict=queries_dict,
                output_folder=args.output_folder,
                qrels_df=qrels,
                use_cache=args.use_cache,
                model=model,
                tokenizer=tokenizer,
                window_size=args.window_size,
                step=args.step
            )
            reranked_mrr = evaluate_mrr_at_k(reranked_df, qrels, k=10, min_rank_threshold=n_ranks_bm25)

            epoch_result = {
                'epoch': epoch_counter,
                'model': checkpoint_identifier,
                'MRR@10': reranked_mrr
            }
            results.append(epoch_result)
            save_results_to_csv([epoch_result], args.output_folder)  # Save each epoch immediately
            
            epoch_counter += 1  # Increment epoch

    elif args.model_name:
        checkpoint_identifier = args.model_name
        safe_identifier = re.sub(r"[^\w\-]+", "_", checkpoint_identifier)
        cache_path = os.path.join(args.output_folder, f"outputs_{safe_identifier}.json")

        model, tokenizer = None, None
        if not args.use_cache:
            print(f"Use_cache=False => forcing a re-run for {checkpoint_identifier}")
            model, tokenizer = load_model_and_tokenizer(args.model_name)
        else:
            if not os.path.exists(cache_path):
                print(f"Use_cache=True but no existing cache for {checkpoint_identifier}, loading model.")
                model, tokenizer = load_model_and_tokenizer(args.model_name)

        reranked_df = rerank_with_cache(
            checkpoint_identifier=checkpoint_identifier,
            n_ranks=n_ranks_bm25,
            queries_dict=queries_dict,
            output_folder=args.output_folder,
            qrels_df=qrels,
            use_cache=args.use_cache,
            model=model,
            tokenizer=tokenizer,
            window_size=args.window_size,
            step=args.step
        )
        reranked_mrr = evaluate_mrr_at_k(reranked_df, qrels, k=10, min_rank_threshold=n_ranks_bm25)

        epoch_result = {
            'epoch': epoch_counter,
            'model': checkpoint_identifier,
            'MRR@10': reranked_mrr
        }
        results.append(epoch_result)
        print(f"Results for {checkpoint_identifier}: {epoch_result}")
        save_results_to_csv([epoch_result], args.output_folder)  # Save immediately

    print("\n=== Evaluation Complete ===")
    final_csv = os.path.join(args.output_folder, "evaluation_results.csv")
    print(f"Results saved to {final_csv}")


if __name__ == "__main__":
    main()