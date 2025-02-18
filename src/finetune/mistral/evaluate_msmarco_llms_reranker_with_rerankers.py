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
from transformers import AutoModelForSequenceClassification
import torch

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
    parser.add_argument("--rerank_with_jina", action="store_true",
                        help="If True, re-rank BM25 candidates with the Jina model.",
                        default=False)
    parser.add_argument("--rank_with_multiqa", action="store_true",
                        help="If True, rank BM25 candidates with the MultiQA model.",
                        default=False)
    
    return parser.parse_args()


def load_jina_reranker():
    model = AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-reranker-v2-base-multilingual',
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to('cuda')  # or 'cpu' if no GPU is available
    model.eval()    
    return model

def load_multiqa_ranker():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    return model

def rank_multiqa(model, query, documents):
    corpus_embeddings = model.encode(documents, convert_to_tensor=True, device='cuda')
    query_embedding = model.encode(query, convert_to_tensor=True, device='cuda')
    scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings, dim=1).cpu()
    return scores.tolist()

def rerank(model, query, documents):
    # Prepare sentence pairs for the model
    sentence_pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(sentence_pairs, max_length=1024)
    return scores


def evaluate_mrr_at_k(rankings, ground_truth, k=10, min_rank_threshold=None):
    """
    Calculate MRR@K (Mean Reciprocal Rank) for a given set of rankings and ground truth relevance.
    
    Parameters:
    - rankings: DataFrame containing ranked results with columns ["qid", "pid", "score"].
    - ground_truth: DataFrame containing relevance labels with columns ["qid", "pid", "relevance"].
    - k: The cutoff rank for MRR computation.
    - min_rank_threshold: If set, queries where all relevant documents are ranked below this threshold will be skipped.
    
    Returns:
    - MRR@K score
    """
    if rankings.empty:
        return 0.0

    mrr = 0.0
    valid_query_count = 0  # This will count only the queries we include
    skipped_queries = 0

    for qid, group in rankings.groupby("qid"):
        relevant_pids = set(ground_truth[ground_truth["qid"] == qid]["pid"].tolist())
        ranked_pids = group["pid"].tolist()[:k]  # Only consider top-k

        # If min_rank_threshold is set, check if the query should be skipped
        if min_rank_threshold is not None:
            full_ranking = group["pid"].tolist()  # Get the full ranking (not just top-k)
            min_relevant_rank = min(
                (full_ranking.index(pid) + 1 for pid in relevant_pids if pid in full_ranking),
                default=float("inf"),
            )
            if min_relevant_rank > min_rank_threshold:
                skipped_queries += 1
                continue  # Skip this query

        found = False
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid in relevant_pids:
                mrr += 1.0 / rank
                found = True
                break

        if found:
            valid_query_count += 1

    print(f"Queries skipped due to min_rank_threshold: {skipped_queries}")
    
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
# Jina caching (and computation)
###############################################################################
def load_or_compute_jina(jina_model, queries_dict, top_k, output_folder, use_cache):
    """
    Re-rank BM25 candidates with the Jina model OR load cached results.
    Also updates queries_dict so that the "topn_docs" DataFrame for each query is re-ordered.
    Returns:
      - jina_reranked_df: DataFrame with columns [qid, pid] in Jina order.
      - queries_dict: updated candidate lists.
    """
    cache_path = os.path.join(output_folder, "jina_cache.json")
    if use_cache and os.path.exists(cache_path):
        print(f"[Cache] Loading Jina re-ranking results from {cache_path}")
        with open(cache_path, "r") as f:
            jina_data = json.load(f)
    else:
        print("Computing Jina re-ranking...")
        jina_data = {}
        for qid, qinfo in tqdm(queries_dict.items(), desc="Jina Re-ranking"):
            query_text = qinfo['query_text']
            topn_docs_df = qinfo['topn_docs']
            documents = topn_docs_df['passage'].tolist()
            pids = topn_docs_df['pid'].tolist()
            scores = rerank(jina_model, query_text, documents)
            # Sort by score descending
            scored_docs = sorted(zip(pids, documents, scores), key=lambda x: x[2], reverse=True)
            jina_data[qid] = [pid for pid, _, _ in scored_docs[:top_k]]
        # Save cache
        with open(cache_path, "w") as f:
            json.dump({str(qid): pids for qid, pids in jina_data.items()}, f, indent=2)

    # Update queries_dict: re-order each query's topn_docs using the Jina order.
    for qid_str, pids in jina_data.items():
        qid = int(qid_str)
        topn_docs_df = queries_dict[qid]['topn_docs']
        # Filter for rows whose pid is in the Jina list
        filtered = topn_docs_df[topn_docs_df['pid'].isin(pids)]
        # Reorder rows to follow the order in pids
        ordered_rows = []
        for pid in pids:
            row = filtered[filtered['pid'] == pid]
            if not row.empty:
                ordered_rows.append(row)
        if ordered_rows:
            queries_dict[qid]['topn_docs'] = pd.concat(ordered_rows)
            
    # Build a DataFrame of Jina ranking for evaluation
    rows = []
    for qid_str, pids in jina_data.items():
        qid = int(qid_str)
        for pid in pids:
            rows.append((qid, pid))
    jina_reranked_df = pd.DataFrame(rows, columns=["qid", "pid"])
    return jina_reranked_df, queries_dict


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


###############################################################################
# Main workflow
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Hard-coded paths to MSMARCO data and qrels
    file_path = "/scratch-shared/tnijdam_ir2_other/MSmarco_data/500top1000.dev"
    qrels_path = "/scratch-shared/tnijdam_ir2_other/MSmarco_data/qrels.dev.tsv"


    # For BM25, we compute a larger candidate set (e.g. 250).
    n_ranks_bm25 = 1000
    # How many top candidates to keep after Jina re-ranking
    top_k_jina = 30
    
    # 1) Load MSMARCO data and filter queries that have at least n_ranks_sliding passages.
    msmarco_data = pd.read_csv(file_path, sep="\t", header=None, names=["qid", "pid", "query", "passage"])
    msmarco_data = msmarco_data.groupby('qid').filter(lambda x: len(x) >= top_k_jina)
    # how many queries
    print("Number of queries:", msmarco_data['qid'].nunique())
    # for testing
    # unique_qids = sorted(msmarco_data['qid'].unique())[:50]
    # msmarco_data = msmarco_data[msmarco_data['qid'].isin(unique_qids)]
    
    # 2) Load qrels.
    qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "0", "pid", "relevance"])

    # 3) BM25: Compute (or load cached) BM25 rankings.
    bm25_rankings, queries_dict = load_or_compute_bm25(msmarco_data, n_ranks_bm25, args.output_folder, args.use_cache)
    bm25_mrr = evaluate_mrr_at_k(bm25_rankings, qrels, k=10)
    print("BM25 MRR@10:", bm25_mrr)

    results = []
    
    results.append({
        "epoch": 0,
        "model": "BM25",
        "MRR@10": bm25_mrr
    })
    
    if args.rerank_with_jina:
        # 4) Jina: Re-rank BM25 candidates (or load cached) with the Jina reranker.
        jina_model = load_jina_reranker()
        jina_reranked_df, queries_dict = load_or_compute_jina(jina_model, queries_dict, top_k_jina, args.output_folder, args.use_cache)
        jina_mrr = evaluate_mrr_at_k(jina_reranked_df, qrels, k=10)
        print("Jina Reranker MRR@10:", jina_mrr)

        results.append({
            "epoch": 0,
            "model": "Jina Reranker",
            "MRR@10": jina_mrr
        })
    
    elif args.rank_with_multiqa:
        print("Ranking BM25 candidates with MultiQA model...")
        # 4) MultiQA: Rank BM25 candidates with the MultiQA model.
        multiqa_model = load_multiqa_ranker()
        multiqa_reranked_scores = []
        for qid, qinfo in tqdm(queries_dict.items(), desc="MultiQA Ranking"):
            query_text = qinfo['query_text']
            topn_docs_df = qinfo['topn_docs']
            scores = rank_multiqa(multiqa_model, query_text, topn_docs_df['passage'].tolist())
            multiqa_reranked_scores.extend(zip([qid] * len(scores), topn_docs_df['pid'], scores))
        multiqa_reranked_df = pd.DataFrame(multiqa_reranked_scores, columns=["qid", "pid", "score"])
        multiqa_reranked_df = multiqa_reranked_df.sort_values(by=["qid", "score"], ascending=[True, False])
        
        # Update queries_dict: re-order each query's topn_docs using the MultiQA order.
        for qid, group in multiqa_reranked_df.groupby("qid"):
            topn_docs_df = queries_dict[qid]['topn_docs']
            ordered_rows = []
            for pid in group["pid"]:
                row = topn_docs_df[topn_docs_df['pid'] == pid]
                if not row.empty:
                    ordered_rows.append(row)
            if ordered_rows:
                queries_dict[qid]['topn_docs'] = pd.concat(ordered_rows)
        
        multiqa_mrr = evaluate_mrr_at_k(multiqa_reranked_df, qrels, k=10)
        print("MultiQA MRR@10:", multiqa_mrr)

        results.append({
            "epoch": 0,
            "model": "MultiQA",
            "MRR@10": multiqa_mrr
        })
        
    intermediate_df = pd.DataFrame(results)
    intermediate_csv = os.path.join(args.output_folder, "evaluation_results_intermediate.csv")
    intermediate_df.to_csv(intermediate_csv, index=False)
    print("Intermediate results saved to", intermediate_csv)

    # 5) Continue with sliding-window re-ranking using the (Jina-updated) queries_dict.
    def extract_checkpoint_number(dirname):
        match = re.search(r'checkpoint-(\d+)', dirname)
        return int(match.group(1)) if match else -1

    # Append further results into the same results list.
    if args.checkpoint_base_dir:
        dirs = [
            d for d in os.listdir(args.checkpoint_base_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.checkpoint_base_dir, d))
        ]
        dirs_sorted = sorted(dirs, key=extract_checkpoint_number)

        for subdir in dirs_sorted:
            full_ckpt_path = os.path.join(args.checkpoint_base_dir, subdir)
            epoch_num = extract_checkpoint_number(subdir)
            checkpoint_identifier = subdir

            model, tokenizer = None, None
            if not args.use_cache:
                print(f"Use_cache=False => forcing a re-run for {checkpoint_identifier}")
                model, tokenizer = load_model_and_tokenizer(full_ckpt_path)
            else:
                safe_id = re.sub(r"[^\w\-]+", "_", subdir)
                cache_path = os.path.join(args.output_folder, f"outputs_{safe_id}.json")
                if not os.path.exists(cache_path):
                    print(f"Use_cache=True but no existing cache for {checkpoint_identifier}, loading model now.")
                    model, tokenizer = load_model_and_tokenizer(full_ckpt_path)

            reranked_df = rerank_with_cache(
                checkpoint_identifier=checkpoint_identifier,
                n_ranks=top_k_jina,
                queries_dict=queries_dict,
                output_folder=args.output_folder,
                qrels_df=qrels,
                use_cache=args.use_cache,
                model=model,
                tokenizer=tokenizer,
                window_size=args.window_size,
                step=args.step
            )
            reranked_mrr = evaluate_mrr_at_k(reranked_df, qrels, k=10)
            results.append({
                'epoch': epoch_num,
                'model': checkpoint_identifier,
                'MRR@10': reranked_mrr
            })
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
            n_ranks=top_k_jina,
            queries_dict=queries_dict,
            output_folder=args.output_folder,
            qrels_df=qrels,
            use_cache=args.use_cache,
            model=model,
            tokenizer=tokenizer,
            window_size=args.window_size,
            step=args.step
        )
        reranked_mrr = evaluate_mrr_at_k(reranked_df, qrels, k=10)
        results.append({
            'epoch': 1,
            'model': checkpoint_identifier,
            'MRR@10': reranked_mrr
        })

    # 6) Save final results to CSV.
    final_df = pd.DataFrame(results)
    final_csv = os.path.join(args.output_folder, "evaluation_results.csv")
    final_df.to_csv(final_csv, index=False)
    print("\n=== Evaluation Complete ===")
    print(final_df)
    print(f"Results saved to {final_csv}")


if __name__ == "__main__":
    main()
