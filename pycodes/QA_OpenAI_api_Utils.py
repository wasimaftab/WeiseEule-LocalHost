#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:31:04 2023

@author: wasim
"""

# import libs
from itertools import groupby
from collections import defaultdict
import tiktoken
# import pdb
# import openai
from openai import AsyncOpenAI
from typing import AsyncGenerator
from openai import OpenAI
# import pinecone
from pinecone import Pinecone
import os
import re
import copy
import pickle
import ast
import pandas as pd
import faiss
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
# import time
pd.set_option('display.max_columns', 500)

# import PMC_downloader_Utils as pmcd
import pycodes.PMC_downloader_Utils as pmcd


# from transformers import BertTokenizer, BertModel
# import torch
# import numpy as np

## Sync client
syncClient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

## Async client
client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
###############################################################################
EMBEDDING_MODEL = "text-embedding-ada-002"
# COMPLETIONS_MODEL = "gpt-3.5-turbo"
# COMPLETIONS_MODEL = "gpt-4"
# MAX_SECTION_LEN = 1000
# MAX_SECTION_LEN = 2500

SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
###############################################################################
# Setup logger
# log_dir = "/home/wasim/Desktop/QA_Bot_Web_App/App/Logs"
log_dir = os.getcwd() + "/Logs"
log_file = "QA_bot.log"
log_file_context = "context.log"
log_file_prompt = "prompt.log"
query_file_prompt = "query.log"

# Create a global variable to log across different places in this file
logger = pmcd.create_logger(log_dir, log_file, 'weiseeule_logger')
# context_logger = pmcd.create_logger(
#     log_dir, log_file_context, 'weiseeule_context_logger')
# prompt_logger = pmcd.create_logger(
#     log_dir, log_file_prompt, 'weiseeule_prompt_logger')

query_logger = pmcd.create_logger(
    log_dir, query_file_prompt, 'weiseeule_query_logger')
###############################################################################

def extract_chunk_paper_id(matches: list) -> dict:
    paper_ids = []
    chunks = []
    vector_ids = []
    for match in matches:
        vector_ids.append(match['id'])
        paper_ids.append(int(match['metadata']['paper_id']))
        chunks.append(match['metadata']['text'])        
    return {'chunks' : chunks, 'paper_ids' : paper_ids, 'vector_ids' : vector_ids}

# def paper_and_vector_ids_from_index(prev_rank:list, matches:list)->dict:
#     vector_id = []
#     paper_id = []
#     for i in prev_rank:
#         vector_id.append(matches[i]['id'])
#         paper_id.append(int(matches[i]['metadata']['paper_id']))
#     return {'paper_id' : paper_id, 'vector_id' : vector_id}

def paper_and_vector_ids_from_index(prev_rank: list, matches: list) -> dict:
    vector_id, paper_id = zip(*[(matches[i]['id'], int(matches[i]['metadata']['paper_id'])) for i in prev_rank])
    return {'paper_id': list(paper_id), 'vector_id': list(vector_id)}

def get_keywords_from_query(text, llm):
    if text.startswith("#"):
        # Find all instances enclosed with double asterisks
        double_star_matches = re.findall(r"\*\*(.*?)\*\*", text)
        
        # Find all instances enclosed with single or double asterisks
        single_star_matches = re.findall(r"(\*{1,2})(.*?)(\1)", text)
        
        # Extract only the matched content from single_star_matches, ignoring the asterisks
        single_star_matches = [match[1] for match in single_star_matches]
        
        # Replace single and double asterisks with an empty string
        query_without_stars = re.sub(r"\*+", "", text.lstrip("#"))
        
        keyword_query = {"code": "success",
                         "filtered_query":query_without_stars, 
                         "search_keywords": single_star_matches, 
                         "primary_keywords":double_star_matches}    
    else: 
        # prompt_init = "Parse the following question for specific biomedical keywords and return them as a Python list assigned to a variable named 'biomedical_keywords'. Provide the response in plain text, without code block formatting or additional commentary: "
        # prompt = prompt_init + text + ' Output only the Python list.'
        prompt_init = "Parse the following question for specific biomedical keywords and return them as a Python list assigned to a variable named 'biomedical_keywords'. Provide the response in plain text and output only the Python list without code block formatting or additional commentary: "
        prompt = prompt_init + text

        # print(f"prompt = {prompt}")
        # openai.api_key = os.getenv("OPENAI_API_KEY") ## old
        try:
            # use sync client for this task
            response = syncClient.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3000,
            top_p=0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            # answer = response["choices"][0]["message"]["content"].strip(" \n") 
            answer = response.choices[0].message.content.strip(" \n")
            
            ## Remove the variable part of the response, leaving only the list
            if "biomedical_keywords = " in answer:
                # Extract just the list part after the equals sign
                list_str = answer.split("biomedical_keywords = ")[1]
                # Convert the string representation of the list to an actual list
                biomedical_keywords_list = ast.literal_eval(list_str)
                keyword_query = {"code": "success", 
                                 "filtered_query": text,
                                 "search_keywords": biomedical_keywords_list,
                                 "primary_keywords": biomedical_keywords_list}
            else:
                # raise ValueError("The expected 'biomedical_keywords' variable assignment is not present in the response.")
                keyword_query = {"code": "failure", 
                                 "msg": "The expected 'biomedical_keywords' variable assignment is not present in the response."}
        except Exception as e:
            # msg = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
            msg = f"Error in entity_extraction: {e}"
            keyword_query = {"code" : "failure", "msg" : msg}
            
    return keyword_query

###############################################################################


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_encoding(model):
    encoding = tiktoken.encoding_for_model(model)
    return encoding
###############################################################################


def remove_values_key_pinecone_res(res):
    data_without_values = []

    for d in res:
        # Deep copy the dictionary to prevent modifying the original object
        dict_representation = copy.deepcopy(d.__dict__)

        # Access the nested _data_store dictionary
        data_store = dict_representation.get('_data_store', {})

        # Remove 'values' key if it exists inside _data_store
        if 'values' in data_store:
            del data_store['values']

        # Create the dictionary with the desired order
        ordered_data_store = {
            'id': data_store.get('id'),
            'metadata': data_store.get('metadata'),
            'score': data_store.get('score')
        }

        # Append to the new list
        data_without_values.append(ordered_data_store)
    return data_without_values


def remove_duplicates(lst):
    unique_elements = []
    for item in lst:
        if item not in unique_elements:
            unique_elements.append(item)
    return unique_elements


def rank_rows(df_non_zero_pk, my_keywords):
    # Extract only the columns containing keywords frequencies
    df_k = df_non_zero_pk.loc[:, my_keywords]

    # Count the number of non-zero elements in each row, excluding the primary_keyword column
    df_k['nnz'] = (df_k != 0).sum(axis=1)

    # Step 4: Concatenate df and df_k hozizontally
    df2_non_zero_pk = pd.concat(
        [df_k, df_non_zero_pk.drop(columns=my_keywords)], axis=1)

    # Sort the dataframe by 'non_zero_count' and 'total_count'
    df2_non_zero_pk = df2_non_zero_pk.sort_values(
        by=['nnz', 'total_count'], ascending=[False, False])

    return df2_non_zero_pk

###############################################################################


def get_keyword_pattern(keyword):
    """
    Create a regex pattern for the given keyword.
    For keywords ending with a number, allow variations with non-word characters before the number.
    """
    if re.search(r'\d$', keyword):
        pattern = re.compile(r'\b' + re.escape(keyword[:-1]) + r'\W*' + re.escape(keyword[-1]) + r'\b', re.IGNORECASE)
    else:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
    return pattern


def rerank_matches_GUI(res: dict, my_keywords: list, keyword_fixed: bool, primary_keywords: list) -> list:
    texts = []
    vector_ids = []
    paper_ids = []
    # chunk_ids = []
    # scores = []
    matches = []
    logger.info(f"Length of res['matches'] inside rerank_matches_GUI() = {len(res['matches'])}")
    for match in res['matches']:
        texts.append(match['metadata']['text'])
        vector_ids.append(match['id'])
        paper_ids.append(match['metadata']['paper_id'])
        # chunk_ids.append(match['metadata']['chunk_id'])
        # scores.append(match['score'])
        matches.append(match)

    # logger.info(f"Length of texts = {len(texts)}")

    # Count the frequency of each keyword in the texts.
    counts = []
    for i, t in enumerate(texts):
        # keyword_counts = {k: t.count(k) for k in my_keywords}
        keyword_counts = {}
        for keyword in my_keywords:
            # pdb.set_trace()
            pattern = get_keyword_pattern(keyword)
            if not isinstance(t, str):
                t = str(t)
            try:
                keyword_counts[keyword] = len(pattern.findall(t))
            except Exception as e:
                logger.info(f"Following exception has occured")
                logger.info(f"Exception: {e}")

        # keyword_counts["total_count"] = sum([keyword_counts[k] for k in my_keywords])  # corrected line
        keyword_counts["total_count"] = sum(keyword_counts.values())
        keyword_counts["text"] = t
        keyword_counts["prev_rank"] = i
        keyword_counts["vector_id"] = vector_ids[i]
        # keyword_counts["score"] = scores[i]
        keyword_counts["paper_id"] = paper_ids[i]
        # keyword_counts["chunk_id"] = chunk_ids[i]
        counts.append(keyword_counts)
        
    # pdb.set_trace()

    df = pd.DataFrame(counts)
    # pdb.set_trace()
    # logger.info(f"logging top 5 rows of counts_df below")
    # logger.info(f"counts_df: {df[:5]}")

    if not keyword_fixed:
        # Create a new column 'non_zero_count'
        df['non_zero_count'] = (df[my_keywords] > 0).sum(axis=1)

        # Sort the dataframe by 'non_zero_count' and 'total_count'
        df_final = df.sort_values(by=['non_zero_count', 'total_count'], ascending=[False, False])
    else:
        # Step 1: Split the df in two separate dfs: one has rows with 0 in all primary_keywords columns and the other has rows with non-zero in any primary_keyword column
        df_zero_pk = df[(df[primary_keywords] == 0).any(axis=1)]
        df_non_zero_pk = df[(df[primary_keywords] != 0).all(axis=1)]

        # Step 2: Call a function using df_non_zero_pk to rank the rows
        df2_non_zero_pk = rank_rows(df_non_zero_pk, my_keywords)

        # Step 3: Call a function using df_zero_pk to rank the rows
        df2_zero_pk = rank_rows(df_zero_pk, my_keywords)

        # Finally join df2_non_zero_pk and df2_zero_pk vertically in same order
        df_final = pd.concat([df2_non_zero_pk, df2_zero_pk], axis=0)

    # Reset the index and create the 'new_rank' column
    df_final = df_final.reset_index(drop=True)
    df_final['new_rank'] = df_final.index

    # Drop unnecessary columns and reorder for clarity
    # df_final = df_final[my_keywords +
    #                     ['total_count', 'prev_rank', 'new_rank', 'score']]
    # df_final = df_final[my_keywords +
    #                     ['total_count', 'prev_rank', 'new_rank', 'paper_id', 'chunk_id']]
    df_final = df_final[my_keywords + ['total_count', 'prev_rank', 'new_rank', 'paper_id', 'vector_id']]


    # Rerank the matches
    idx = df_final.prev_rank.to_list()
    reranked_matches = [matches[i] for i in idx]

    # Extract indices of matches with zero keyword frequency
    no_keywords_indices = list(df_final[df_final.total_count == 0].index)

    # Now delete matches with zero keyword frequency
    if no_keywords_indices:
        del reranked_matches[-len(no_keywords_indices):]

    # logger.info(f"matches before re-ranking = {len(matches)}\nmatches after re-ranking = {len(reranked_matches)}\n")

    # Also remove those texts records from dataframe
    df_final = df_final.drop(index=no_keywords_indices)
    return df_final
###############################################################################

def rerank_matches(res:dict, my_keywords: list, keyword_fixed: bool, primary_keywords: list) -> list:
    texts = []
    # paper_ids = []
    scores = []
    matches = []
    for match in res['matches']:
        texts.append(match['metadata']['text'])
        # paper_ids.append(match['metadata']['paper_id'])
        scores.append(match['score'])
        matches.append(match)

    # Count the frequency of each keyword in the texts.
    counts = []
    for i, t in enumerate(texts):
        keyword_counts = {k: t.count(k) for k in my_keywords}
        # pdb.set_trace()
        keyword_counts["total_count"] = sum(
            [keyword_counts[k] for k in my_keywords])  # corrected line
        keyword_counts["text"] = t
        keyword_counts["prev_rank"] = i
        keyword_counts["score"] = scores[i]
        counts.append(keyword_counts)

    df = pd.DataFrame(counts)
    # pdb.set_trace()

    if not keyword_fixed:
        # Create a new column 'non_zero_count'
        df['non_zero_count'] = (df[my_keywords] > 0).sum(axis=1)

        # Sort the dataframe by 'non_zero_count' and 'total_count'
        df_final = df.sort_values(
            by=['non_zero_count', 'total_count'], ascending=[False, False])
    else:
        # Step 1: Split the df in two separate dfs: one has rows with 0 in all primary_keywords columns and the other has rows with non-zero in any primary_keyword column
        df_zero_pk = df[(df[primary_keywords] == 0).any(axis=1)]
        df_non_zero_pk = df[(df[primary_keywords] != 0).all(axis=1)]

        # Step 2: Call a function using df_non_zero_pk to rank the rows
        df2_non_zero_pk = rank_rows(df_non_zero_pk, my_keywords)

        # Step 3: Call a function using df_zero_pk to rank the rows
        df2_zero_pk = rank_rows(df_zero_pk, my_keywords)

        # Finally join df2_non_zero_pk and df2_zero_pk vertically in same order
        df_final = pd.concat([df2_non_zero_pk, df2_zero_pk], axis=0)

    # Reset the index and create the 'new_rank' column
    df_final = df_final.reset_index(drop=True)
    df_final['new_rank'] = df_final.index

    # Drop unnecessary columns and reorder for clarity
    df_final = df_final[my_keywords +
                        ['total_count', 'prev_rank', 'new_rank', 'score']]

    # pdb.set_trace()

    # Rerank the matches
    idx = df_final.prev_rank.to_list()
    reranked_matches = [matches[i] for i in idx]

    # Extract indices of matches with zero keyword frequency
    no_keywords_indices = list(df_final[df_final.total_count == 0].index)

    # Now delete matches with zero keyword frequency
    if no_keywords_indices:
        del reranked_matches[-len(no_keywords_indices):]

    print(
        f"matches before re-ranking = {len(matches)}\nmatches after re-ranking = {len(reranked_matches)}\n")

    # Also remove those texts records from dataframe
    df_final = df_final.drop(index=no_keywords_indices)

    # print(df_final.iloc[:3,:])
    # print(df_final)
    top_n_rows = 50
    print(df_final[:top_n_rows])

    # pdb.set_trace()

    resume = input(
        f"\nTake a look at the keyword frequency table (top {top_n_rows} rows) printed above, enter 1 if you want to resume, otherewise press any key to exit = ")

    res = {}
    if resume == '1':
        res['code'] = 'success'
        select_vecs = input(
            "\nWould you like to select vectors? enter 1 if yes, otherewise press any key to continue with default set of vectors = ")
        if select_vecs == '1':
            idx_vecs = get_list_of_values(input_str="vector index")

            # in case duplicate indices entered
            idx_vecs = list(set(idx_vecs))
            res['output'] = [reranked_matches[i] for i in idx_vecs]
        else:
            res['output'] = reranked_matches

    else:
        res['code'] = 'failure'
        res['output'] = "User chosen to terminate excecution"

    return res

## Mark keywords with HTML tag
def mark_keywords(text, keywords):
    """
    Replaces each occurrence of every keyword in the text with <mark>keyword</mark>.
    
    Parameters:
    - text (str): The input text.
    - keywords (list): List of keywords to mark in the text.
    
    Returns:
    - str: The text with keywords marked.
    """
    # Use a case-insensitive search to match all occurrences regardless of case
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(f"<mark>{keyword}</mark>", text)
    return text


# v2
def construct_prompt(question: str, 
                     template: str, 
                     model: str, 
                     rerank_flag: bool,
                     reranked_matches: dict,
                     search_keywords: list) -> dict:    
    model_lengths = {
    "gpt-4": 6500,
    "gpt-4-1106-preview": 8000,  # Replace with actual value
    "gpt-4o": 8000,
    "gpt-3.5-turbo-1106": 10000  # Replace with actual value
    }
    # Set a default length if the model is not in the dictionary
    default_length = 3000

    # Get the MAX_SECTION_LEN based on the model
    MAX_SECTION_LEN = model_lengths.get(model, default_length)

    list_of_dict = []
    chosen_sections = []
    chosen_sections_len = 0
    # MAX_SECTION_LEN = 6500 if model == "gpt-4" else 3000
    separator_len = num_tokens_from_string(SEPARATOR, model)
    citation_list = []
    prompt_citation = {}
    context = []
    # pdb.set_trace()
    for match in reranked_matches['matches']:
        try:
            ## When rerank_flag == TRUE, mark keywords with html tag for highlighting in UI
            if rerank_flag:
                context.append(f"{mark_keywords(match['metadata']['text'], search_keywords)} [Paper ID: {str(int(match['metadata']['paper_id']))}]<br>***<br>")
            else:
                context.append(f"{match['metadata']['text']} [Paper ID: {str(int(match['metadata']['paper_id']))}]<br>***<br>")
            
            list_of_dict.append(match["metadata"])
            text = match['metadata']['text'].replace("\n", " ")
            chosen_sections.append(SEPARATOR + text)
            chosen_sections_len += num_tokens_from_string(
                text, model) + separator_len
            citation_list.append(f"{match['metadata']['citation']} [Paper ID: {str(int(match['metadata']['paper_id']))}]")
            if chosen_sections_len > MAX_SECTION_LEN:
                break
        except:
            output = "Error appending context/citation in Pinecone query results, did you select correct namespace?"
            prompt_citation['output'] = output
            prompt_citation['code'] = "failure"
            return prompt_citation

    # Header without template
    header = """Answer the question as truthfully as possible using the provided context, do not include duplicate statements in the answer and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    # ## Header with template
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # use the information in the template to map ortholog proteins across different species
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""

    # ## Header with template-V2
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""
    prompt_citation['prompt'] = header + \
        "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    prompt_citation['citation'] = remove_duplicates(citation_list)
    prompt_citation['code'] = 'success'

    # Join chunks in context list into a single chunk
    prompt_citation['context'] = "\n".join([el for el in context])
    # pdb.set_trace()
    return prompt_citation
    # return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def construct_prompt_per_paper(question: str, 
                               template: str,
                               rerank_flag: bool,
                               search_keywords: list,
                               match: dict) -> str:
    """
    Fetch relevant 
    """
    # list_of_dict = []
    chosen_sections = []
    prompt_citation = {}
    context = []
    try:
        # pdb.set_trace()
        # context.append(f"{match['score']:.2f}: {match['metadata']['text']}\n\nRef:- {match['metadata']['citation']}\n\n")

        # context.append(f"{match['score']:.2f}: {mark_keywords(match['metadata']['text'], search_keywords)}\n\nRef:- {match['metadata']['citation']}\n\n")
        # list_of_dict.append(match["metadata"])

        ## When rerank_flag == TRUE, mark keywords with html tag for highlighting in UI
        if rerank_flag:
            context.append(f"{mark_keywords(match['metadata']['text'], search_keywords)} [Paper ID: {str(int(match['metadata']['paper_id']))}]<br>***<br>")
        else:
            context.append(f"{match['metadata']['text']} [Paper ID: {str(int(match['metadata']['paper_id']))}]<br>***<br>")        
        chosen_sections.append(SEPARATOR + match['metadata']['text'])
    except:
        output = "Error appending context/citation in Pinecone query results, did you select correct namespace?"
        prompt_citation['output'] = output
        prompt_citation['code'] = "failure"
        return prompt_citation

    # Header without template
    header = """Answer the question as truthfully as possible using the provided context, do not include duplicate statements in the answer and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    # ## Header with template
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # use the information in the template to map ortholog proteins across different species
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""

    # ## Header with template-V2
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""
    prompt_citation['prompt'] = header + \
        "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    prompt_citation['citation'] = match['metadata']['citation'] + \
        " [Paper ID: " + str(int(match['metadata']['paper_id'])) + "]"
    prompt_citation['code'] = 'success'

    # Join chunks in context list into a single chunk
    prompt_citation['context'] = "\n".join([el for el in context])

    return prompt_citation

###############################################################################


def get_list_of_keywords():
    # Initialize an empty list
    user_list = []

    # Get input from the user until they enter an empty string
    while True:
        element = input("Enter a keyword (or press Enter to finish): ")
        if element == "":
            break
        user_list.append(element.lower())

    print("Your list of keywords:", user_list)
    return user_list


def get_list_of_values(input_str):
    # Initialize an empty list
    user_list = []

    # Get input from the user until they enter an empty string
    while True:
        # element = input("Enter an keyword (or press Enter to finish): ")
        element = input(f"Enter a {input_str} (or press Enter to finish): ")
        if element == "":
            break
        elif not 0 <= int(element) <= 14:
            print('The vector index must lie beween 0 and 14, both inclusive, try again')
            continue
        else:
            user_list.append(int(element))

    print(f"Your list of {input_str}: ", user_list)
    return user_list


def order_chunks_of_same_paper(L):
    # Group by 'paper_id'
    groups = defaultdict(list)
    for item in L:
        groups[item['metadata']['paper_id']].append(item)

    # Sort each group by 'chunk_id'
    sorted_groups = {k: sorted(
        v, key=lambda x: x['metadata']['chunk_id']) for k, v in groups.items()}

    # Flatten the sorted groups into a single list
    L_sorted = [item for group in sorted_groups.values() for item in group]
    return L_sorted


def concatenate_chunks_of_same_paper(L):
    L_sorted = order_chunks_of_same_paper(L)

    # Group by 'paper_id'
    L_grouped = []
    for key, group in groupby(L_sorted, key=lambda x: x['metadata']['paper_id']):
        L_grouped.append(list(group))

    # Concatenate the chunks of the same paper and make them as one record
    L_concatenated = []
    for group in L_grouped:
        # concatenated_text = ' '.join(item['metadata']['text'] for item in group)
        temp = []
        for item in group:
            text = item['metadata']['text']
            # if not text.endswith("."):
            #     text += "."
            # temp.append(text)
            temp.append(f"{text}<br>***<br>")
            concatenated_text = ' '.join(temp)
        average_score = sum(item['score'] for item in group) / len(group)

        # pdb.set_trace()
        concatenated_record = {
            # 'id': group[0]['id'],
            'metadata': {
                'citation': group[0]['metadata']['citation'],
                'paper_id': group[0]['metadata']['paper_id'],
                'text': concatenated_text,
            },
            'score': average_score,
            # 'values': group[0]['values'],
        }

        L_concatenated.append(concatenated_record)

    # Sort by 'paper_id'
    L_final = sorted(L_concatenated, key=lambda x: x['metadata']['paper_id'])
    # print(L_final)
    return (L_final)
###############################################################################


def extract_chunks_from_one_paper(res: dict, paper_id: int) -> dict:
    matched_chunks = {}
    matched_chunks['matches'] = []
    for match in res['matches']:
        if paper_id == match['metadata']['paper_id']:
            matched_chunks['matches'].append(match)
    logger.info(f"Length of res['matches'] inside extract_chunks_from_one_paper() = {len(res['matches'])}")
    return matched_chunks
    # return order_chunks_of_same_paper(matched_chunks)
###############################################################################


def load_all_chunks_in_namespace(namespace):
    try:
        with open('Pickled_vecs/' + namespace + '_chunks.pkl', 'rb') as f:
            chunks_in_namespace = pickle.load(f)
        msg = 'chunks loaded successfully'
        load_res = {"code": 'success', "msg": msg, "chunks_in_namespace": chunks_in_namespace}
    except Exception as e:
        msg = f'chunks loading failed! Exception: {e}'
        load_res = {"code": 'failure', "msg": msg}
    return load_res
###############################################################################
###############################################################################

async def chat_answer_per_paper(res: dict,
                                COMPLETIONS_API_PARAMS: dict,
                                query: str,
                                template: str,
                                rerank_flag: bool,
                                search_keywords: list) -> AsyncGenerator[dict, None]:
    reranked_matches_grouped = concatenate_chunks_of_same_paper(res['matches'])

    chunk_count = len(reranked_matches_grouped)
    for match in reranked_matches_grouped:
            prompt_citation = construct_prompt_per_paper(query,
                                                        template,
                                                        rerank_flag,
                                                        search_keywords,
                                                        match)
            if prompt_citation['code'] == 'failure':
                logger.error(prompt_citation['output'])
                
            else:
                prompt = prompt_citation['prompt']
                # logger.info(f"prompt: {prompt}")
                citation = prompt_citation['citation']
                # logger.info(f"citation: {citation}\n*********")
                context = prompt_citation['context']

            try:
                logger.info(f"Inside chat_answer_per_paper() before calling response = await client.chat.completions.create(...)")
                response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}], 
                                                        **COMPLETIONS_API_PARAMS)
                
                ## Working
                async for chunk in response:
                    if chunk.choices[0].finish_reason is None:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield {'content': content}

                # Send citation and context after response is complete
                if chunk_count == 1:
                    yield {'citation': citation, 'last_context': context}
                else:
                    yield {'citation': citation, 'context': context}
                
                ## Decrease count by 1 after processing each chunk 
                logger.info(f"chunk_count = {chunk_count}")   
                chunk_count -= 1
            except Exception as e:
                error_message = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(error_message)
                yield {"error": error_message}

    logger.info(f"chunk_count = {chunk_count}")  


async def chat_answer_one_paper (res: dict,
                                COMPLETIONS_API_PARAMS: dict,
                                query: str,
                                template: str,
                                COMPLETIONS_MODEL: str,
                                rerank_flag: bool,
                                search_keywords: list,
                                paper_id: int) -> AsyncGenerator[dict, None]:
    reranked_matches = extract_chunks_from_one_paper(res, paper_id)
    if not reranked_matches['matches']:
        error_message = f"The dict key 'matches' has zero values! check paper_id. Provided paper_id = {paper_id}"
        yield {"error": error_message}
    else:    
        prompt_citation = construct_prompt(query,
                                        template,
                                        COMPLETIONS_MODEL,
                                        rerank_flag,
                                        reranked_matches,
                                        search_keywords)
        if prompt_citation['code'] == 'failure':
            error_message = f"prompt constuction error: {prompt_citation['output']}"
            yield {"error": error_message}
            
        else:
            prompt = prompt_citation['prompt']
            citation = prompt_citation['citation']
            context = prompt_citation['context']
            
            try:
                response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}], 
                                                        **COMPLETIONS_API_PARAMS)
                
                ## Working
                async for chunk in response:
                    if chunk.choices[0].finish_reason is None:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield {'content': content}
                
                # Send citation and context after response is complete
                yield {'citation': citation, 'context': context}    
            except Exception as e:
                error_message = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(error_message)
                yield {"error": error_message}


async def chat_answer_all_papers(res: dict,
                                COMPLETIONS_API_PARAMS: dict,
                                query: str,
                                template: str,
                                COMPLETIONS_MODEL: str,
                                rerank_flag: bool,
                                search_keywords: list) -> AsyncGenerator[dict, None]:
    reranked_matches = {'matches': order_chunks_of_same_paper(res['matches'])}
    if not reranked_matches['matches']:
        error_message = f"reranked_matches dict is empty"
        yield {"error": error_message}
    else:    
        prompt_citation = construct_prompt(query,
                                        template,
                                        COMPLETIONS_MODEL,
                                        rerank_flag,
                                        reranked_matches,
                                        search_keywords)
        if prompt_citation['code'] == 'failure':
            error_message = f"prompt constuction error: {prompt_citation['output']}"
            yield {"error": error_message}
            
        else:
            prompt = prompt_citation['prompt']
            citation = prompt_citation['citation']
            context = prompt_citation['context']

            ## Create the numbered list with '<br>' tags
            numbered_citation = '<br><br>'.join([f'{i+1}. {citation}' for i, citation in enumerate(citation)])
            
            try:
                response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}], 
                                                        **COMPLETIONS_API_PARAMS)
                
                ## Working
                async for chunk in response:
                    if chunk.choices[0].finish_reason is None:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield {'content': content}
                
                # Send citation and context after response is complete
                yield {'citation': numbered_citation, 'context': context}    
            except Exception as e:
                error_message = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(error_message)
                yield {"error": error_message}

###############################################################################

async def scientific_qa_bot_GUI_stream(COMPLETIONS_MODEL, 
                                       temp, 
                                       namespace, 
                                       query, 
                                       template, 
                                       search_keywords,
                                       primary_keywords,
                                       embedd_model,
                                       paper_id=-1, 
                                       keyword_fixed=False,
                                       answer_per_paper=False, 
                                       chunks_from_one_paper=False,
                                       rerank_flag=False, 
                                       top_k=10):
    ## Call chat completion API
    COMPLETIONS_API_PARAMS = {
        "temperature": temp,
        "max_tokens": 1200 if COMPLETIONS_MODEL.startswith("gpt-4") else 800,
        "model": COMPLETIONS_MODEL,
        "stream": True
    }

    if rerank_flag:
        ## Handle the exception when search_keywords and primary_keywords are empty
        if keyword_fixed and not bool(primary_keywords):
            yield {'error': f'keyword_fixed is {bool(keyword_fixed)} but primary_keywords list is empty'}
        elif rerank_flag and not bool(search_keywords):
            yield {'error': f'rerank_flag is {bool(rerank_flag)} but search_keywords list is empty'}
        
        ## Load all vectors in a namespace
        load_res = load_all_chunks_in_namespace(namespace)
        if load_res['code'] == 'failure':
            yield {"error": load_res['msg']}
        else:
            res = {"matches": load_res['chunks_in_namespace']}  
        
        ## Check if selected namespace is valid
        if not res["matches"]:
            output = "The dict key 'matches' has zero values, did you select correct namespace?"
            logger.error(output)
            yield {"error": output}
        
        ## Extract chunks from a single paper and handle the exception for invalid paper_id, becaus that will generate empty res!
        logger.info(f"chunks_from_one_paper = {chunks_from_one_paper}")
        if chunks_from_one_paper:
            res = extract_chunks_from_one_paper(res, paper_id)

        ## Re-rank pinecone chunks based on keyword frequency
        ranked_df = rerank_matches_GUI(res, search_keywords, keyword_fixed, primary_keywords)
         
        if ranked_df.empty:
            yield {"error": "Empty keyword frequency table! check for typo in your keyword(s) and also make sure you are searching in revelant namespace"}
        else:
            ## return all top_n chunks that contain where the primary_keywords atleast appear
            top_n_rows = 10
            # df_final = ranked_df[(ranked_df[primary_keywords] !=0).all(axis=1)] 
            # df_final = df_final[:top_n_rows] if df_final.shape[0] > top_n_rows else df_final
            
            ## my modification
            if query.startswith("##"):
                df_final = ranked_df[(ranked_df[primary_keywords] !=0).all(axis=1)] 
                df_final = df_final[:top_n_rows] if df_final.shape[0] > top_n_rows else df_final
            else:
                df_final = ranked_df[:top_n_rows] if ranked_df.shape[0] > top_n_rows else ranked_df
            logger.info(f"Inside scientific_qa_bot_GUI_stream() and length of ranked_df.shape[0] = {ranked_df.shape[0]}")


            ## Convert DataFrame to JSON
            if df_final.empty:
                yield {'error': f'Re-ranked df is empty'}
            else:
                df_json = df_final.to_json(orient='split')
            # logger.info('df_json:')
            # logger.info(df_json)

            ## Write logic to extract chunks using vector_ids      
            vector_ids = df_final.iloc[:,-1].to_list()
            filtered_chunks = [d for d in res["matches"] if d['id'] in set(vector_ids)]
            
            ## Reorder the filtered chunks
            filtered_chunks.sort(key=lambda x: vector_ids.index(x['id']))
            
            ## update res with filtered chunks only
            res['matches'] = filtered_chunks
            logger.info(f"Inside scientific_qa_bot_GUI_stream() and length of res['matches'] = {len(res['matches'])}")

            ## Here write the logic of prompt and answer gereration
            if answer_per_paper:
                async for content in chat_answer_per_paper(res,
                                                        COMPLETIONS_API_PARAMS,
                                                        query,
                                                        template,
                                                        rerank_flag,
                                                        search_keywords):
                    yield content
                pass
            elif chunks_from_one_paper:
                async for content in chat_answer_one_paper(res,
                                      COMPLETIONS_API_PARAMS,
                                      query,
                                      template,
                                      COMPLETIONS_MODEL,
                                      rerank_flag,
                                      search_keywords,
                                      paper_id):
                    yield content
                pass
            else:
                async for content in chat_answer_all_papers(res,
                                COMPLETIONS_API_PARAMS,
                                query,
                                template,
                                COMPLETIONS_MODEL,
                                rerank_flag,
                                search_keywords):
                    yield content
                # pass

            ## Send DataFrame as JSON to WebSocket Endpoint           
            yield {'df': df_json}      

    else:    
        query_logger.info(f"Q:- {query}")
        index = pmcd.get_pinecone_index()
        xq = pmcd.get_text_embeddings('query', [query], embedd_model, openai_vec_len=1536)
        # xq = pmcd.get_text_embeddings('chunk', [query], embedd_model, openai_vec_len=1536)


        if embedd_model == 'openai':
            res = index.query(vector=[xq], top_k=top_k, include_metadata=True, namespace=namespace)
        elif embedd_model == 'biobert':
            res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)
        elif embedd_model == 'MedCPT':
            res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)

        if not res["matches"]:
            error_message = f"The dict key 'matches' has zero values, did you select the correct namespace?"
            yield {"error": error_message}
            
        res['matches'] = order_chunks_of_same_paper(res['matches'])
        prompt_citation = construct_prompt(query, 
                                        template, 
                                        COMPLETIONS_MODEL, 
                                        rerank_flag, 
                                        res, 
                                        search_keywords)
        if prompt_citation['code'] == 'failure':
            # print(f"prompt constuction error: {prompt_citation['output']}")
            error_message = f"prompt constuction error: {prompt_citation['output']}"
            yield {"error": error_message}
        else:
            prompt = prompt_citation['prompt']
            citation = prompt_citation['citation']
            context = prompt_citation['context']

            ## Create the numbered list with '<br>' tags
            numbered_citation = '<br><br>'.join([f'{i+1}. {citation}' for i, citation in enumerate(citation)])

            COMPLETIONS_API_PARAMS = {
                "temperature": temp,
                "max_tokens": 1200 if COMPLETIONS_MODEL.startswith("gpt-4") else 800,
                "model": COMPLETIONS_MODEL,
                "stream": True
            }
            try:
                response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}], 
                                                        **COMPLETIONS_API_PARAMS)
                
                async for chunk in response:
                    if chunk.choices[0].finish_reason is None:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield {'content': content}
                            # print(content)

                # Send citation and context after response is complete
                yield {'citation': numbered_citation, 'context': context}
            except Exception as e:
                error_message = f"Response error from OpenAI! Actual exception is: {e}"
                yield {"error": error_message}


###############################################################################

def search_necessary_files(pm_embed_chunk_id, MedCPT_folder):
    embeds_chunk_file = "embeds_chunk_" + pm_embed_chunk_id + ".npy"
    pmids_chunk_file = "pmids_chunk_" + pm_embed_chunk_id + ".json"
    pubmed_chunk_file = "pubmed_chunk_" + pm_embed_chunk_id + ".json"
    
    logger.info(f"Inside async def search_necessary_files(...) and pwd: {os.getcwd()}")
    logger.info(f"Inside async def search_necessary_files(...) and MedCPT_folder: {MedCPT_folder}")
    
    if not os.path.isfile(MedCPT_folder + "/" + embeds_chunk_file):
        error = """Embeeding file does not exist, make sure you have downloaded 
                the chunk and supplied the number at the end of query after #. For example:
                 Cerebral small vessel disease #34"""
        return {'status':'error', 'message':error}
    
    if not os.path.isfile(MedCPT_folder + "/" + pmids_chunk_file):
        error = """PMID file does not exist, make sure you have downloaded 
            the chunk and supplied the number at the end of query after #. For example:
             Cerebral small vessel disease #34"""
        return {'status':'error', 'message':error}

    if not os.path.isfile(MedCPT_folder + "/" + pubmed_chunk_file):
        error = """PMID2CONTENT file does not exist, make sure you have downloaded 
                the chunk and supplied the number at the end of query after #. For example:
                 Cerebral small vessel disease #34"""
        return {'status':'error', 'message':error}
    
    return {'status':'success', 'message':'All the required files are found'}
##---------------------------------------------------------------------------##

## latest version: With chunk id
async def search_PubMed(query):
    logger.info(f"Inside async def search_PubMed(query) and pwd: {os.getcwd()}")
    pattern = r"#\s*\d+"
    if not re.search(pattern, query):
        error_message = """The query does not contain a number preceded by '#'.
               A valid query looks like: Cerebral small vessel disease #34
               The number at the end points to a pubmed embedding chunk which 
               should be downloaded first"""
        logger.info('query does not contain #')
        yield {'error':error_message}
        return
    
    try:
        # Extract chunk id from query
        pm_embed_chunk_id = query.split("#")[1].replace(" ", "")
    except IndexError:
        error_message = """The query does not contain a valid number preceded by '#'.
               A valid query looks like: Cerebral small vessel disease #34
               The number at the end points to a pubmed embedding chunk which 
               should be downloaded first"""
        logger.info('Error in extracting chunk ID')
        yield {"error": error_message}
        return

    MedCPT_folder = "MedCPT_Embeddings"

    # Check if necessary files exist
    res = search_necessary_files(pm_embed_chunk_id, MedCPT_folder)
    if res['status'] == 'error':
        yield {'error': res['message']}
        return
    else:
        logger.info(f"Inside async def search_PubMed(query) and {res['message']}")
       
    # ## Example: loading a chunk hardcoded in the file name
    # pubmed_embeds = np.array(np.load("/home/wasim/Desktop/MedCPT_Embeddings/embeds_chunk_36.npy"))
    # pmids = json.load(open("/home/wasim/Desktop/MedCPT_Embeddings/pmids_chunk_36.json"))
    # pmid2content = json.load(open("/home/wasim/Desktop/MedCPT_Embeddings/pubmed_chunk_36.json"))
    
    ## Example: loading a chunk using variable in the file name
    # embeds_chunk_file = MedCPT_folder + "/embeds_chunk_" + pm_embed_chunk_id + ".npy"
    # pmids_chunk_file = MedCPT_folder + "/pmids_chunk_" + pm_embed_chunk_id + ".json"
    # pubmed_chunk_file = MedCPT_folder + "/pubmed_chunk_" + pm_embed_chunk_id + ".json"

    embeds_chunk_file = os.path.join(MedCPT_folder, f"embeds_chunk_{pm_embed_chunk_id}.npy")
    pmids_chunk_file = os.path.join(MedCPT_folder, f"pmids_chunk_{pm_embed_chunk_id}.json")
    pubmed_chunk_file = os.path.join(MedCPT_folder, f"pubmed_chunk_{pm_embed_chunk_id}.json")
    
    pubmed_embeds = np.array(np.load(embeds_chunk_file))
    pmids = json.load(open(pmids_chunk_file))
    pmid2content = json.load(open(pubmed_chunk_file))
    
    ## Index vectors
    index = faiss.IndexFlatIP(768)
    index.add(pubmed_embeds)
    
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    queries = []
    logger.info(f'Actual query is being searched is: {query.split("#")[0].strip()}')

    queries.append(query.split("#")[0].strip()) # need to remove the chunk ID from query before embedding
    # queries.append(query)

    with torch.no_grad():
        # tokenize the queries
        encoded = tokenizer(
            queries, 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )

        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model(**encoded).last_hidden_state[:, 0, :]
    
        # search the Faiss index
        scores, inds = index.search(embeds, k=10)
    
    PMID = []
    SCORE = []
    TITLE = []
    for idx, query in enumerate(queries):
        for score, ind in zip(scores[idx], inds[idx]):
            PMID.append(f"<a href='https://pubmed.ncbi.nlm.nih.gov/{pmids[ind]}/'>{pmids[ind]}</a>")
            SCORE.append(round(float(score), 2))
            TITLE.append(pmid2content[pmids[ind]]['t'])
    df = pd.DataFrame(list(zip(PMID, TITLE, SCORE)), columns=['PMID', 'TITLE', 'SCORE'])
    df_json = df.to_json(orient='split')
    yield {'df': df_json, 'last_content': "Done"}

###############################################################################

## latest version: With chunk id
async def process_pmids_and_summarize(pmids:list, llm:str, query:str):
    logger.info(f"Inside process_pmids_and_summarize(...)")
    logger.info(f"type(pmids) = {type(pmids)}")
    logger.info(f"pmids = {pmids}")
    logger.info(f"llm = {llm}")
    logger.info(f"query = {query}")

    ## Small example
    # pmid2content = json.load(open("./MedCPT_Embeddings/first_100_pmid2content_from_34.json"))

    ## Real example
    # pmid2content = json.load(open("/home/wasim/Desktop/MedCPT_Embeddings/pubmed_chunk_36.json"))
    
    MedCPT_folder = "MedCPT_Embeddings"
    pm_embed_chunk_id = query.split("#")[1].replace(" ", "")
    pubmed_chunk_file = MedCPT_folder + "/pubmed_chunk_" + pm_embed_chunk_id + ".json"
    pmid2content = json.load(open(pubmed_chunk_file))
    
    ## remove chunk number from query for sign in prompt
    query = query.split("#")[0].rstrip()
    logger.info(f"After removing chunk number query = {query}")
    
    # selected_abstracts = [abstracts_dict[pmid] for pmid in pmids if pmid in abstracts_dict]
    # selected_abstracts = [abstracts_dict[pmid] for pmid in pmids if pmid in abstracts_dict and abstracts_dict[pmid]]
    selected_abstracts = []
    for pmid in pmids:
        if pmid in pmid2content:
            if pmid2content[pmid]['a']:
                selected_abstracts.append(pmid2content[pmid]['a'])

    if not selected_abstracts:
        error_message = "No abstracts found for the provided PMIDs."
        yield {"error": error_message}


    # Create a prompt for GPT-4 to summarize the abstracts
    context = ""
    # context = "<br>"
    for abstract in selected_abstracts:
        context += f"{abstract}\n"
        # context += f"{abstract}<br>"

    header = "Summarize the context provided below for a scientist, focusing on the intent of the given query. Do not include duplicate statements in the summary."
    # prompt = header + query + context
    # prompt = header + '<br>Query:<br>' + query + '<br><br>Context:<br>' + context
    prompt = header + "\n\nQuery:\n" + query + "\n\nContext:\n" + context

    ## Call OpenAI API to get the summary
    COMPLETIONS_API_PARAMS = {
        "temperature": 1.0,
        "max_tokens": 1200 if llm.startswith("gpt-4") else 800,
        "model": llm,
        "stream": True
    }
    try:
        response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
                                                **COMPLETIONS_API_PARAMS)
        async for chunk in response:
            if chunk.choices[0].finish_reason is None:
                content = chunk.choices[0].delta.content
                if content:
                    yield {'summary': content}
        yield {'end_summary' : 'Done!'}
    except Exception as e:
        error_message = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
        yield {"error": error_message}

###############################################################################

async def summarize_selected_content(text, llm):
    # header = "Summarize the context provided below for a scientist. Do not include duplicate statements in the summary."
    header = "Clearly explain the context provided below to a scientist. Do not include duplicate statements in the summary."

    prompt = header + "\n\nContext:\n" + text


    ## Call OpenAI API to get the summary
    COMPLETIONS_API_PARAMS = {
        "temperature": 1.0,
        "max_tokens": 1200 if llm.startswith("gpt-4") else 800,
        "model": llm,
        "stream": True
    }
    try:
        response = await client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
                                                **COMPLETIONS_API_PARAMS)
        async for chunk in response:
            if chunk.choices[0].finish_reason is None:
                content = chunk.choices[0].delta.content
                if content:
                    yield {'summary': content}
        yield {'end_summary' : 'Done!'}
    except Exception as e:
            error_message = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
            yield {"error": error_message}

###############################################################################

def get_embeddings(query, top_k, namespace):
    index = pmcd.get_pinecone_index()
    xq = client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding

    # Query vector Db for chunks that matches with query embeeding
    res = index.query(vector=[xq], top_k=top_k, include_metadata=True,
                      include_values=True, namespace=namespace)
    emb = []
    for match in res['matches']:
        emb.append(match['values'])

    embeddings = {}
    embeddings['query'] = xq
    embeddings['top_matches'] = emb
    return embeddings
###############################################################################

def summarize_text(text, prompt_init, llm):
    # prompt = "Summarize this for a second-grade student:\n\n" + text
    prompt = prompt_init + text
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        # response = openai.ChatCompletion.create(
        response = client.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # answer = response["choices"][0]["message"]["content"].strip(" \n")
        answer = response.choices[0].message.content.strip(" \n")
        return {"code": "success", "msg": answer}
    except Exception as e:
        msg = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
        return {"code": "error", "msg": msg}
###############################################################################

def extract_table(answer):
    # Use regex to find table content
    match = re.search(r'\|(.|\n)+\|', answer)
    if match:
        return match.group(0), answer[:match.start()].strip().replace("\\n", "<br>")
    return '', ''


def markdown_to_html_table(md_table, description):
    # Start with the description
    html_output = "<p>{}</p>\n".format(description)

    # Split the table into lines
    lines = md_table.split("\\n")

    # Start the table
    html_output += "<table>\n"

    # Parse each line
    for i, line in enumerate(lines):
        # Split the line into cells
        cells = [cell.strip() for cell in line.split("|") if cell]

        # Start a table row
        html_output += "<tr>\n"

        # Parse each cell
        for cell in cells:
            # Check if it's the header row
            if i == 0:
                html_output += "<th>{}</th>\n".format(cell)
            else:
                html_output += "<td>{}</td>\n".format(cell)

        # End the table row
        html_output += "</tr>\n"

    # End the table
    html_output += "</table>"

    return html_output
###############################################################################
###############################################################################
