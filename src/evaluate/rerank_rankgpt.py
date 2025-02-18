from external.rankgpt.rank_gpt_utils import permutation_pipeline, permutation_pipeline_few_shot 

def calc_preffered_rankgpt(doc1, doc2, q1, q2, model_name="gpt-4o", model=None, api_key=None):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q1, q2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model: string containing the model name for the new reranker

    Returns:
        A dictionary containing each query (q1 or q2) and the pairwise correctness score,
        the model (if applicable), and a similarity score (None in this case).
    """
    # Prepare the inputs for the reranker
    item1 = {
        'query': q1,
        'hits': [
            {'content': doc1},
            {'content': doc2}
        ]
    }
    item2 = {
        'query': q2,
        'hits': [
            {'content': doc1},
            {'content': doc2}
        ]
    }

    # Run the reranker
    new_item1 = permutation_pipeline(item1, rank_start=0, rank_end=2, model_name=model_name, api_key=api_key)
    new_item2 = permutation_pipeline(item2, rank_start=0, rank_end=2, model_name=model_name, api_key=api_key)

    # Print ranked outputs for debugging
    # print(new_item1)
    # print("-------------")
    # print(new_item2)

    # Results dictionary
    results = {}

    # Check pairwise correctness
    num_correct = 0
    # For q1, doc1 should be ranked higher (i.e., come first in hits)
    if new_item1["hits"][0]["content"] == doc1:
        num_correct += 1
    # For q2, doc2 should be ranked higher (i.e., come first in hits)
    if new_item2["hits"][0]["content"] == doc2:
        num_correct += 1

    # Add results for each query and overall score
    results["q1"] = [hit["content"] for hit in new_item1["hits"]]
    results["q2"] = [hit["content"] for hit in new_item2["hits"]]
    results["score"] = num_correct / 2  # Normalize to a range of [0, 1]

    # Compatibility with the original function's return structure
    model = model  # Replace with actual model object if needed
    similarity_score = None

    return results, model, similarity_score


def calc_preffered_rankgpt_few_shot(train_triplets_df, doc1, doc2, q1, q2, model_name="gpt-4o", model=None, api_key=None, x_shot=1):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q1, q2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model: string containing the model name for the new reranker

    Returns:
        A dictionary containing each query (q1 or q2) and the pairwise correctness score,
        the model (if applicable), and a similarity score (None in this case).
    """
    # Prepare the inputs for the reranker
    item1 = {
        'query': q1,
        'hits': [
            {'content': doc1},
            {'content': doc2}
        ]
    }
    item2 = {
        'query': q2,
        'hits': [
            {'content': doc1},
            {'content': doc2}
        ]
    }

    # Run the reranker
    new_item1 = permutation_pipeline_few_shot(train_triplets_df, item1, rank_start=0, rank_end=2, model_name=model_name, api_key=api_key, x_shot=x_shot)
    new_item2 = permutation_pipeline_few_shot(train_triplets_df, item2, rank_start=0, rank_end=2, model_name=model_name, api_key=api_key, x_shot=x_shot)

    # Print ranked outputs for debugging
    # print(new_item1)
    # print("-------------")
    # print(new_item2)

    # Results dictionary
    results = {}

    # Check pairwise correctness
    num_correct = 0
    # For q1, doc1 should be ranked higher (i.e., come first in hits)
    if new_item1["hits"][0]["content"] == doc1:
        num_correct += 1
    # For q2, doc2 should be ranked higher (i.e., come first in hits)
    if new_item2["hits"][0]["content"] == doc2:
        num_correct += 1

    # Add results for each query and overall score
    results["q1"] = [hit["content"] for hit in new_item1["hits"]]
    results["q2"] = [hit["content"] for hit in new_item2["hits"]]
    results["score"] = num_correct / 2  # Normalize to a range of [0, 1]

    # Compatibility with the original function's return structure
    model = model  # Replace with actual model object if needed
    similarity_score = None

    return results, model, similarity_score
