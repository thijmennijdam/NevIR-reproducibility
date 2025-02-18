from sentence_transformers import (
    SentenceTransformer,
    util,
    CrossEncoder,
    models,
)
import torch
import argparse

#TODO: move to utils
def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    # np.random.seed(seed)  # Set seed for NumPy
    # random.seed(seed)  # Set seed for Python's random module
    torch.manual_seed(seed)  # Set seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch on current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for PyTorch on all GPUs

    # Allow non-deterministic behavior for performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def fine_calc_preferred_dense(doc1, doc2, q1, q2, path_to_finetuned_model, load_finetune, base_model="multi-qa-mpnet-base-dot-v1"):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        model: the preloaded model, if caching
        return_similarity: whether to return the similarity score or not

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair

    """
    print("Loading model...")
    print("Loading finetuned model: ", load_finetune)
    print(f"Path to finetuned model: {path_to_finetuned_model}")
    if load_finetune == str2bool(load_finetune):
        embedder = SentenceTransformer(path_to_finetuned_model)  
    else:
        embedder = SentenceTransformer(base_model)

    corpus = [doc1, doc2]
    queries = [q1, q2]
    results = {}
    num_correct = 0
    doc_sim = None

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    doc_sim = torch.nn.functional.cosine_similarity(corpus_embeddings[0].unsqueeze(0), corpus_embeddings[1].unsqueeze(0))

    for idx, query in enumerate(queries):
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        scores = util.dot_score(query_embedding, corpus_embeddings)[0].cpu()
        results[f"q{idx+1}"] = scores.tolist()
        should_be_higher = scores[idx]
        should_be_lower = scores[0] if idx != 0 else scores[1]
        if should_be_higher > should_be_lower:
            num_correct += 1
    model = embedder

    print("Loading finetuned model: ", load_finetune)
    print(f"Path to finetuned model: {path_to_finetuned_model}")
    results["score"] = num_correct / 2
    return results, model, doc_sim


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-f",
#         "--file",
#         help="whether to load a file and if so, the path",
#         type=str,
#         default=None,
#     )
#     parser.add_argument(
#         "-d1",
#         "--doc1",
#         help="doc1 if loading from command line",
#         type=str,
#         default=None,
#     )
#     parser.add_argument(
#         "-d2",
#         "--doc2",
#         help="doc2 if loading from command line",
#         type=str,
#         default=None,
#     )
#     parser.add_argument(
#         "-q1",
#         "--q1",
#         help="q1 if loading from command line",
#         type=str,
#         default=None,
#     )
#     parser.add_argument(
#         "-q2",
#         "--q2",
#         help="q1 if loading from command line",
#         type=str,
#         default=None,
#     )
#     parser.add_argument(
#         "-m",
#         "--model_name",
#         help="the model to use, if not loading from file",
#         type=str,
#         default="dpr",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for reproducibility.",
#     )
#     parser.add_argument(
#         "--path_to_finetuned_model",
#         type=str,
#         default="/home/scur0987/project/IR2/results_finetune/finetune_NevIR/sentence-transformers_multi-qa-mpnet-base-dot-v1/model",
#         help="Path to the finetuned model",
#     )
#     args = parser.parse_args()
#     if not args.file and (
#         not args.doc1
#         or not args.doc2
#         or not args.q1
#         or not args.q2
#         or args.model
#     ):
#         print(
#             "Error: need either a file path or the input args (d1, d2, q1, q2, model)"
#         )
#     elif args.file:
#         _set_seed(args.seed)
#         print("Loading from file...")
#         df = pd.read_csv(args.file)
#         scores = []
#         sim_scores = []
#         model = None
#         global_model_name = args.model_name if args.model_name is not None else None
#         for (idx, row) in tqdm.tqdm(df.iterrows(), total=len(df)):
#             model_name = global_model_name if global_model_name is not None else row["model_name"]
#             results, model, sim_score = calc_preferred_dense(
#                     row["doc1"],
#                     row["doc2"],
#                     row["q1"],
#                     row["q2"],
#                     path_to_finetuned_model = args.path_to_finetuned_model,
#                     model=model
#                 )
#             scores.append(
#                 results["score"]
#             )
#             sim_scores.append(sim_score.item())
#         print((np.array(scores) == 1).mean())
#         if global_model_name is not None:
#             model_name = global_model_name.split("/")[-3] + "-" + global_model_name.split("/")[-1]
            
#         with open(args.file.replace("csv", f"{model_name.replace('/', '_')}.results"), "w") as f:
#             f.write(json.dumps({
#                 "scores": scores,
#                 "sim_scores": np.mean(sim_scores),
#                 "sim_scores_std": np.std(sim_scores),
#                 "paired_accuracy": (np.array(scores) == 1).mean(),
#                 "model": model_name
#             }))
#         print(args.file.replace("csv", f"{model_name.replace('/', '_')}.results"))
#     else:
#         print("Loading from args...")
#         print(
#             calc_preferred_dense(
#                 args.doc1, args.doc2, args.q1, args.q2, args.model_name
#             )[0]
#         )
