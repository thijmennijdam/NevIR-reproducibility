from transformers import pipeline
import torch
import random

def construct_messages_mistral(query, doc1, doc2):
    """Constructs a message list for Llama Instruct to simulate role-based prompting."""
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    # Prefix Prompt
    messages = [
        {"role": "system", 
         "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
        {"role": "user", 
         "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", 
         "content": "Okay, please provide the passages."},
        {"role": "user", 
         "content": (
             f"{passages}\n\n"
             f"Search Query: {query}.\n"
             "Rank the passages above based on their relevance to the search query. "
             "The passages should be listed in descending order using identifiers. "
             "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
             "Only respond with the ranking results, do not say any word or explain."
         )}
    ]

    return messages

def construct_messages_mistral_few_shot(train_triplets_df, query, doc1, doc2, x_shot=1):
    """Constructs a message list for Llama Instruct to simulate role-based prompting."""
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    def sample_train_triplets(train_df, x_shot):
        """Randomly sample num_samples triplets from the training data."""
        # take the last num_samples rows of the dataframe into new df   
        return train_df.tail(x_shot).values.tolist()
    
    samples = sample_train_triplets(train_triplets_df, x_shot)
    # print(len(samples))
    # Prefix Prompt
    message_list = [
        {"role": "system", 
         "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."}]
    
    for sample in samples:  
        train_query, pos, neg = sample
        
        message_list.append({'role': 'user', 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {train_query}."})
        message_list.append({'role': 'assistant', 'content': 'Okay, please provide the passages.'})
        passages_samples = [pos, neg]

        random.shuffle(passages_samples)
        # get rank of positive sample
        pos_rank = passages_samples.index(pos) + 1
        neg_rank = passages_samples.index(neg) + 1
        passages_train = f"[1] {passages_samples[0]}\n[2] {passages_samples[1]}"

        message_list.append({"role": "user", 
            "content": 
                f"{passages_train}\n\n"
                f"Search Query: {query}.\n"
                "Rank the passages above based on their relevance to the search query. "
                "The passages should be listed in descending order using identifiers. "
                "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
                "Only respond with the ranking results, do not say any word or explain."
            })
        message_list.append({'role': 'assistant', 'content': f'[{pos_rank}] > [{neg_rank}]'})

    message_list.append({'role': 'user', 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {train_query}."})
    message_list.append({'role': 'assistant', 'content': 'Okay, please provide the passages.'})
    message_list.append({"role": "user", 
         "content": 
             f"{passages}\n\n"
             f"Search Query: {query}.\n"
             "Rank the passages above based on their relevance to the search query. "
             "The passages should be listed in descending order using identifiers. "
             "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
             "Only respond with the ranking results, do not say any word or explain."
         })
    # print(message_list)
    return message_list

def construct_messages_llama(query, doc1, doc2):
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    prefix_messages = [{'role': 'system',
                    'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                    {'role': 'user',
                    'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                    {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    # User's input with passages
    user_passages = {"role": "user", "content": passages}
    
    # Post Prompt
    post_prompt = {"role": "user", 
                   "content": (
                       f"Search Query: {query}.\n"
                       f"Rank the {num} passages above based on their relevance to the search query. "
                       "The passages should be listed in descending order using identifiers. "
                       "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
                       "Only respond with the ranking results, do not say any word or explain."
                   )}
    
    return prefix_messages + [user_passages, post_prompt]

def construct_messages_grit(query, doc1, doc2):
    """Constructs a message list for GritLM to simulate role-based prompting without system messages."""
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    # Construct the user instructions
    messages = [
        {"role": "user", 
         "content": (
             f"I will provide you with {num} passages, each indicated by a number identifier []. \nRank the passages based on their relevance to the query: {query}."
             f"{passages}\n\n"
             f"Search Query: {query}.\n"
             "Rank the passages above based on their relevance to the search query. "
             "The passages should be listed in descending order using identifiers. "
             "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
             "Only respond with the ranking results, do not say any word or explain."
         )}
    ]

    return messages

def calc_llms_rerankers(doc1, doc2, q1, q2, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", generator=None):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q1, q2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        pipeline: the preloaded pipeline, if caching

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair
    """
    queries = [q1, q2]
    passages = [doc1, doc2]
    results = {}
    num_correct = 0

    for idx, query in enumerate(queries):
        print("MODEL NAME: ", model_name)
        # Construct messages for Llama Instruct
        if "llama" in model_name.lower():
            messages = construct_messages_llama(query, doc1, doc2)
        elif "mistral" in model_name.lower() or "qwen" in model_name.lower():
            messages = construct_messages_mistral_few_shot(query, doc1, doc2)
        elif "grit" in model_name.lower():
            messages = construct_messages_grit(query, doc1, doc2)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Generate the response
        if generator is None:
            generator = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16)
        
        
        generation = generator(
            messages,
            do_sample=False,
            temperature=0,
            top_p=1,
            max_new_tokens=10
        )
        # response = pipeline(messages)[0]["generated_text"][-1]
        response = generation[0]["generated_text"][-1]

        try:
            # Extract the content part if the response is a dict
            if isinstance(response, dict):
                response = response.get("content", "")

            ranks = response.strip().split(" > ")
            first_rank = int(ranks[0].strip("[]"))
            second_rank = int(ranks[1].strip("[]"))
        except (ValueError, IndexError):
            raise ValueError(f"Unexpected output format: {response}")

        scores = [0, 0]  # Default scores
        scores[first_rank - 1] = 1

        results[f"q{idx + 1}"] = scores

        # Check correctness
        if scores[idx] > scores[1 - idx]:
            num_correct += 1

    results["score"] = num_correct / 2
    return results, generator, None

# Example usage
if __name__ == "__main__":
    doc1 = "Machine learning is a field of AI focused on algorithms."
    doc2 = "Machine learning is a cooking technique for making desserts."
    q1 = "What is machine learning?"
    q2 = "How do you make desserts?"

    results = calc_llms_rerankers(doc1, doc2, q1, q2)
    print(results)


def construct_messages_mistral(query, doc1, doc2):
    """Constructs a message list for Llama Instruct to simulate role-based prompting."""
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    # Prefix Prompt
    messages = [
        {"role": "system", 
         "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
        {"role": "user", 
         "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", 
         "content": "Okay, please provide the passages."},
        {"role": "user", 
         "content": (
             f"{passages}\n\n"
             f"Search Query: {query}.\n"
             "Rank the passages above based on their relevance to the search query. "
             "The passages should be listed in descending order using identifiers. "
             "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
             "Only respond with the ranking results, do not say any word or explain."
         )}
    ]

    return messages


def construct_messages_llama(query, doc1, doc2):
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    prefix_messages = [{'role': 'system',
                    'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                    {'role': 'user',
                    'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                    {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    # User's input with passages
    user_passages = {"role": "user", "content": passages}
    
    # Post Prompt
    post_prompt = {"role": "user", 
                   "content": (
                       f"Search Query: {query}.\n"
                       f"Rank the {num} passages above based on their relevance to the search query. "
                       "The passages should be listed in descending order using identifiers. "
                       "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
                       "Only respond with the ranking results, do not say any word or explain."
                   )}
    
    return prefix_messages + [user_passages, post_prompt]

def construct_messages_grit(query, doc1, doc2):
    """Constructs a message list for GritLM to simulate role-based prompting without system messages."""
    num = 2  # Number of passages
    passages = f"[1] {doc1}\n[2] {doc2}"

    # Construct the user instructions
    messages = [
        {"role": "user", 
         "content": (
             f"I will provide you with {num} passages, each indicated by a number identifier []. \nRank the passages based on their relevance to the query: {query}."
             f"{passages}\n\n"
             f"Search Query: {query}.\n"
             "Rank the passages above based on their relevance to the search query. "
             "The passages should be listed in descending order using identifiers. "
             "The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. "
             "Only respond with the ranking results, do not say any word or explain."
         )}
    ]

    return messages

def calc_llms_rerankers_few_shot(train_df, doc1, doc2, q1, q2, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", generator=None, x_shot=1):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q1, q2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        pipeline: the preloaded pipeline, if caching

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair
    """
    queries = [q1, q2]
    passages = [doc1, doc2]
    results = {}
    num_correct = 0

    for idx, query in enumerate(queries):
        print("MODEL NAME: ", model_name)
        # Construct messages for Llama Instruct
        if "llama" in model_name.lower():
            messages = construct_messages_llama(query, doc1, doc2)
        elif "mistral" in model_name.lower() or "qwen" in model_name.lower():
            messages = construct_messages_mistral_few_shot(train_df, query, doc1, doc2, x_shot=x_shot)
        elif "grit" in model_name.lower():
            messages = construct_messages_grit(query, doc1, doc2)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Generate the response
        if generator is None:
            generator = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16)
        
        
        generation = generator(
            messages,
            do_sample=False,
            temperature=0,
            top_p=1,
            max_new_tokens=10
        )
        # response = pipeline(messages)[0]["generated_text"][-1]
        response = generation[0]["generated_text"][-1]

        try:
            # Extract the content part if the response is a dict
            if isinstance(response, dict):
                response = response.get("content", "")

            ranks = response.strip().split(" > ")
            first_rank = int(ranks[0].strip("[]"))
            second_rank = int(ranks[1].strip("[]"))
        except (ValueError, IndexError):
            raise ValueError(f"Unexpected output format: {response}")

        scores = [0, 0]  # Default scores
        scores[first_rank - 1] = 1

        results[f"q{idx + 1}"] = scores

        # Check correctness
        if scores[idx] > scores[1 - idx]:
            num_correct += 1

    results["score"] = num_correct / 2
    return results, generator, None

# Example usage
if __name__ == "__main__":
    doc1 = "Machine learning is a field of AI focused on algorithms."
    doc2 = "Machine learning is a cooking technique for making desserts."
    q1 = "What is machine learning?"
    q2 = "How do you make desserts?"

    results = calc_llms_rerankers(doc1, doc2, q1, q2)
    print(results)