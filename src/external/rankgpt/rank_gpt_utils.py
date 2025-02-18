import copy
from tqdm import tqdm
import time
import json
import random   


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion

def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_prefix_prompt_few_shot(train_triplets_df, query, num, x_shot):
    max_length = 300

    def sample_train_triplets(train_df, x_shot):
        """Randomly sample num_samples triplets from the training data."""
        # return train_df.head(num_samples).values.tolist()
        return train_df.tail(x_shot).values.tolist()
    
    samples = sample_train_triplets(train_triplets_df, x_shot)

    message_list = [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."}]
    
    for sample in samples:
        train_query, pos, neg = sample
        
        message_list.append({'role': 'user', 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {train_query}."})
        message_list.append({'role': 'assistant', 'content': 'Okay, please provide the passages.'})
        passages = [pos, neg]

        random.shuffle(passages)
        # get rank of positive sample
        pos_rank = passages.index(pos) + 1
        neg_rank = passages.index(neg) + 1

        for idx, hit in enumerate(passages):
            rank = idx + 1
            content = hit
            content = content.replace('Title: Content: ', '')
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = ' '.join(content.split()[:int(max_length)])
            message_list.append({'role': 'user', 'content': f"[{rank}] {content}"})
            message_list.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
        message_list.append({'role': 'user', 'content': get_post_prompt(train_query, num)})
        message_list.append({'role': 'assistant', 'content': f'[{pos_rank}] > [{neg_rank}]'})

    message_list.append({'role': 'user', 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."})
    message_list.append({'role': 'assistant', 'content': 'Okay, please provide the passages.'})
    
    return message_list


def get_post_prompt(query, num):
    if num == 4:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > [] > [], e.g., [1] > [2] > [3] > [4] or perhaps [2] > [1] > [3] > [4] etc. Only response the ranking results, do not say any word or explain."
    if num == 2:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2] or [2] > [1]. Only response the ranking results, do not say any word or explain."

def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo'):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def create_permutation_instruction_few_shot(train_triplets_df, item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', x_shot=1):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt_few_shot(train_triplets_df, query, num, x_shot)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages, api_key=None, model_name="gpt-3.5-turbo"):
    if 'gpt' in model_name or 'o' in model_name:
        Client = OpenaiClient
    else:
        raise ValueError('Model not supported')

    agent = Client(api_key)
    if 'gpt' in model_name:
        response = agent.chat(model=model_name, messages=messages, temperature=0, return_text=True)
    elif 'o' in model_name:
        response = agent.chat(model=model_name, messages=messages, return_text=True)
    return response

def run_mistral(messages, model, tokenizer):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    generated = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, do_sample=False, temperature=0.0)
    decoded_output = tokenizer.batch_decode(generated, skip_special_tokens=False)
    permutation = extract_ranking(decoded_output[0])
    # print(permutation)
    return permutation

import re

def extract_ranking(response: str):
    """
    Given a response string, this function finds the final output (i.e., the text
    after the last [/INST] tag) and extracts a ranking string of the form:
        [3] > [1] > [4] > [2]
    (the ranking can have any length). It returns the list of ranking numbers in order.
    
    If no valid ranking is found in the final output, an empty list is returned.
    """
    # Find the last occurrence of the closing [/INST] tag.
    last_close_inst = response.rfind('[/INST]')
    if last_close_inst != -1:
        # Consider only the text after the last [/INST]
        final_output = response[last_close_inst + len('[/INST]'):]
    else:
        # If no [/INST] tag is found, use the entire response.
        final_output = response

    # Define a regex pattern to capture a sequence like "[digit] > [digit] > ...".
    pattern = r'(\[\d+\](?:\s*>\s*\[\d+\])+)'
    
    # Search for the ranking pattern in the final output.
    match = re.search(pattern, final_output)
    if match:
        ranking_str = match.group(1)
        digits = re.findall(r'\[(\d+)\]', ranking_str)

        formatted_ranking = " ".join(digits)  # Convert to space-separated string
        return formatted_ranking  # ✅ Matches what clean_response() expects
    else:
        print("No valid ranking found.")
        return ''

def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None, model=None, tokenizer=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
    # print(model_name )
    if 'mistral' in model_name or 'checkpoint' in model_name:
        permutation = run_mistral(messages, model, tokenizer)
    else:
        permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item

def permutation_pipeline_few_shot(train_triplets_df, item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None, x_shot=1):
    # print("permutation_pipeline_few_shot")
    messages = create_permutation_instruction_few_shot(train_triplets_df, item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name, x_shot=x_shot)  # chan
    
    # print messages
    # print("messages")

    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    # print("permutation received")
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    # print("permutation applied")
    return item

def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None, model=None, tokenizer=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key, model=model, tokenizer=tokenizer)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item

def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

