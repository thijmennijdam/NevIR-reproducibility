import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from external.rankgpt.rank_gpt_utils import sliding_windows
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# STEP 1: Load in MSMARCO data and qrels
file_path = "/scratch-shared/tnijdam_ir2/data/MSmarco_data/500top1000.dev"
qrels_path = "/scratch-shared/tnijdam_ir2/data/MSmarco_data/qrels.dev.tsv"
n_ranks = 30

msmarco_data = pd.read_csv(file_path, sep="\t", header=None, names=["qid", "pid", "query", "passage"])
qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "0", "pid", "relevance"])

# STEP 2: Process the first 10 queries
query_ids = msmarco_data['qid'].unique()[:10]

bm25_rankings = []  # To store BM25 rankings for all queries
reranked_rankings = []  # To store reranked results for all queries

# STEP 3: Load the model and tokenizer
# checkpoint_path = "unsloth/mistral-7b-instruct-v0.3"
checkpoint_path = "/scratch-shared/tnijdam_ir2/weights/mistral/NevIR-conversational/checkpoint-2380"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    # max_seq_length=2048,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="mistral",
    map_eos_token=True,
)
    
for qid in tqdm(query_ids):
    # Get data for the current query
    query_data = msmarco_data[msmarco_data['qid'] == qid].copy()

    # Initialize BM25 for the passages
    bm25 = BM25Okapi(query_data['passage'].str.split().tolist())

    # Compute BM25 scores for the query
    query_text = query_data['query'].iloc[0]
    bm25_scores = bm25.get_scores(query_text.split())

    # Add BM25 scores to the DataFrame
    query_data['bm25_score'] = bm25_scores

    # Sort and take the top n_ranks documents
    topn_documents = query_data.sort_values(by='bm25_score', ascending=False).head(n_ranks)

    # Save BM25 rankings for evaluation
    bm25_rankings.append(topn_documents[["qid", "pid"]])

    # Prepare data for the reranker
    df = pd.DataFrame({
        'query': [query_text],  # Single query
        'hits': [[{'pid': pid, 'content': passage} for pid, passage in zip(topn_documents['pid'], topn_documents['passage'])]]
    })

    # Rerank using the sliding windows
    for _, item in df.iterrows():
        reranked_item = sliding_windows(
            item, rank_start=0, rank_end=n_ranks, window_size=4, step=2, 
            model_name=checkpoint_path, model=model, tokenizer=tokenizer
        )

        # Extract reranked passages and their PIDs
        reranked_passages = reranked_item["hits"]
        reranked_rankings.append(pd.DataFrame({
            'qid': [qid] * len(reranked_passages),
            'pid': [passage['pid'] for passage in reranked_passages]  # Extract `pid` from the reranked output
        }))

# Combine BM25 rankings into a single DataFrame
bm25_rankings = pd.concat(bm25_rankings).reset_index(drop=True)

# Combine reranked rankings into a single DataFrame
reranked_rankings = pd.concat(reranked_rankings).reset_index(drop=True)

# STEP 4: Define MRR@10 Evaluation Function
def evaluate_mrr_at_k(rankings, ground_truth, k=10):
    """
    Calculates the MRR@K (Mean Reciprocal Rank at K) for a given set of rankings and ground truth relevance.
    rankings: DataFrame containing query ID and ranked passage IDs.
    ground_truth: DataFrame containing query ID and relevant passage IDs.
    k: The cutoff rank (default is 10).
    """
    mrr = 0.0
    for qid, group in rankings.groupby("qid"):
        # Extract relevant passage IDs for this query
        relevant_pids = set(ground_truth[ground_truth["qid"] == qid]["pid"].tolist())
        
        # Extract ranked passage IDs for this query
        ranked_pids = group["pid"].tolist()[:k]  # Take top-k only
        
        # Find the first relevant passage in the ranked list
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid in relevant_pids:
                mrr += 1 / rank
                break
    
    # Normalize by the number of queries
    num_queries = rankings["qid"].nunique()
    return mrr / num_queries

# STEP 5: Compute MRR@10 for BM25 and Reranked Results
bm25_mrr_at_10 = evaluate_mrr_at_k(bm25_rankings, qrels, k=10)
reranked_mrr_at_10 = evaluate_mrr_at_k(reranked_rankings, qrels, k=10)

print(f"BM25 MRR@10: {bm25_mrr_at_10:.4f}")
print(f"Reranked MRR@10: {reranked_mrr_at_10:.4f}")