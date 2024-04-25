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
import pdb
import openai
import pinecone
import os
import re
import copy
import pickle
import ast
import pandas as pd
pd.set_option('display.max_columns', 500)

import PMC_downloader_Utils as pmcd

# Comment this line before running GUI
# import pycodes.PMC_downloader_Utils as pmcd


# from transformers import BertTokenizer, BertModel
# import torch
# import numpy as np
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
        prompt_init = "Parse the following question for specific biomedical keywords and return them as a Python list assigned to a variable named 'biomedical_keywords'. Provide the response in plain text, without code block formatting or additional commentary: "
        prompt = prompt_init + text + ' Output only the Python list.'
        # print(f"prompt = {prompt}")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            response = openai.ChatCompletion.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3000,
            top_p=0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            answer = response["choices"][0]["message"]["content"].strip(" \n") 
            
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
    for match in res['matches']:
        texts.append(match['metadata']['text'])
        vector_ids.append(match['id'])
        paper_ids.append(match['metadata']['paper_id'])
        # chunk_ids.append(match['metadata']['chunk_id'])
        # scores.append(match['score'])
        matches.append(match)

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
    # df_final = df_final[my_keywords +
    #                     ['total_count', 'prev_rank', 'new_rank', 'score']]
    # df_final = df_final[my_keywords +
    #                     ['total_count', 'prev_rank', 'new_rank', 'paper_id', 'chunk_id']]
    df_final = df_final[my_keywords +
                        ['total_count', 'prev_rank', 'new_rank', 'paper_id', 'vector_id']]


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

def rerank_matches(res: pinecone.QueryResponse, my_keywords: list, keyword_fixed: bool, primary_keywords: list) -> list:
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
                     reranked_matches: dict,
                     search_keywords: list) -> dict:    
    model_lengths = {
    "gpt-4": 6500,
    "gpt-4-1106-preview": 8000,  # Replace with actual value
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
    for match in reranked_matches['matches']:
        try:
            # pdb.set_trace()
            # context.append(f"{match['metadata']['text']}")
            # context.append(f"{match['metadata']['text']}\n***\n***")
            # context.append(f"{match['metadata']['text']}<br>***<br>***")
            # context.append(f"{match['metadata']['text']}<br>***<br>")

            ## mark keywords with html tag for highlighting in UI
            # context.append(f"{mark_keywords(match['metadata']['text'], search_keywords)}<br>***<br>")
            context.append(f"{mark_keywords(match['metadata']['text'], search_keywords)} [Paper ID: {str(int(match['metadata']['paper_id']))}]<br>***<br>")
            
            list_of_dict.append(match["metadata"])
            text = match['metadata']['text'].replace("\n", " ")
            chosen_sections.append(SEPARATOR + text)
            chosen_sections_len += num_tokens_from_string(
                text, model) + separator_len
            # citation_list.append(match['metadata']['citation'])
            citation_list.append(f"{match['metadata']['citation']} [Paper ID: {str(int(match['metadata']['paper_id']))}]")
            if chosen_sections_len > MAX_SECTION_LEN:
                break
            # pdb.set_trace()
        except:
            # pdb.set_trace()
            output = "Error appending context/citation in Pinecone query results, did you select correct namespace?"
            prompt_citation['output'] = output
            prompt_citation['code'] = "failure"
            return prompt_citation

    # Log context
    # context_logger.info(
    #     "\n##---------------##\n".join(context) + "\n>>>> END CONTEXT >>>>\n\n")

    # Save the metadata
    # df_citation_text = pd.DataFrame(list_of_dict)
    # excel_file_path = "other_knowledge_sources/" + "_".join(question.split(" ")[:5])+".xlsx"
    # df_citation_text.to_excel(excel_file_path, index=False)

    # Header without template
    header = """Answer the question as truthfully as possible using the provided context, do not include duplicate statements in the answer and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    # ## Header with template
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # use the information in the template to map ortholog proteins across different species
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""

    # ## Header with template-V2
    # header = f"""Answer the question as truthfully as possible using the provided context and template,
    # and if the answer is not contained within the text below, say "I don't know."\n\nTemplate:\n{template}\n\nContext:\n"""
    # pdb.set_trace()
    prompt_citation['prompt'] = header + \
        "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    # prompt_citation['citation'] = remove_duplicates(citation_list)
    prompt_citation['citation'] = remove_duplicates(citation_list)
    prompt_citation['code'] = 'success'
    # Join chunks in context list into a single chunk
    prompt_citation['context'] = "\n".join([el for el in context])
    # pdb.set_trace()
    return prompt_citation
    # return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def construct_prompt_per_paper(question: str, 
                               template: str, 
                               model: str, 
                               match: dict,
                               search_keywords: list) -> str:
    """
    Fetch relevant 
    """
    list_of_dict = []
    chosen_sections = []
    prompt_citation = {}
    context = []
    try:
        # pdb.set_trace()
        context.append(
            f"{match['score']:.2f}: {match['metadata']['text']}\n\nRef:- {match['metadata']['citation']}\n\n")

        # context.append(f"{match['score']:.2f}: {mark_keywords(match['metadata']['text'], search_keywords)}\n\nRef:- {match['metadata']['citation']}\n\n")
        list_of_dict.append(match["metadata"])
        chosen_sections.append(SEPARATOR + match['metadata']['text'])
    except:
        output = "Error appending context/citation in Pinecone query results, did you select correct namespace?"
        prompt_citation['output'] = output
        prompt_citation['code'] = "failure"
        return prompt_citation

    # Log context
    # context_logger.info(
    #     "\n##---------------##\n".join(context) + "\n>>>> END CONTEXT >>>>\n\n")

    # # Save the metadata
    # df_citation_text = pd.DataFrame(list_of_dict)
    # excel_file_path = "other_knowledge_sources/" + "_".join(question.split(" ")[:5])+".xlsx"
    # df_citation_text.to_excel(excel_file_path, index=False)

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
    # pdb.set_trace()
    # prompt_citation['citation'] = match['metadata']['citation']
    prompt_citation['citation'] = match['metadata']['citation'] + \
        " [Paper ID: " + str(int(match['metadata']['paper_id'])) + "]"
    prompt_citation['code'] = 'success'
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
    return matched_chunks
    # return order_chunks_of_same_paper(matched_chunks)
###############################################################################


# v2: Context followed by citation
def scientific_qa_bot(llm,
                      temp,
                      namespace,
                      query,
                      template,
                      search_keywords,
                      primary_keywords,
                      embedd_model,
                      paper_id=-1,
                      answer_per_paper=False,
                      chunks_from_one_paper=False,
                      keyword_fixed=False,
                      top_k=5):
    
    ## Log query
    query_logger.info(f"Q:- {query}")

    # search_keywords = get_list_of_keywords() ## uncomment this line when running in production

    index = pmcd.get_pinecone_index()

    # ## compute OpenAI embedding
    # xq = openai.Embedding.create(input=query, engine=EMBEDDING_MODEL)['data'][0]['embedding']

    # compute BioBERT embedding
    xq = pmcd.get_text_embeddings([query], embedd_model, openai_vec_len=1536)
    # xq = get_text_embeddings([query], embedd_model, openai_vec_len=1536)

    # Query vector Db for chunks that matches with query embeeding
    start = pmcd.time.time()  # time modiule was imported in pmcd, so use from there
    # if embedd_model == 'openai':
    #     res = index.query(vector=[xq], top_k=top_k, include_metadata=True, namespace=namespace) ## make xq a list of list before calling
    # elif embedd_model == 'biobert':
    #     res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace) ## xq is already list of list
    # xq is already list of list, required for pincone query function
    res = index.query(vector=xq, top_k=top_k,
                      include_metadata=True, namespace=namespace)
    end = pmcd.time.time()
    print(
        f"Time taken to fetch top {len(res['matches'])} matching chunks = {round(end - start, 3)} seconds")
    # pdb.set_trace()

    if not res["matches"]:
        output = "The dict key 'matches' has zero values, did you select correct namespace?"
        logger.error(output)
        return {"code": "failure", "msg": output}

    # Extract chunks from a single paper
    if chunks_from_one_paper:
        res = extract_chunks_from_one_paper(res, paper_id)
        answer_per_paper = False
        # pdb.set_trace()

    # Re-rank pinecone chunks based on keyword frequency
    rerank_flag = input(
        "Enter 1 if you want to re-rank chunks before prompt generation, otherewise press any key = ")
    if rerank_flag == '1':
        res = rerank_matches(res, search_keywords,
                             keyword_fixed, primary_keywords)
        if res['code'] == 'failure':
            logger.error(res['output'])
            return {"code": res['code'], "msg": res['output']}
        else:
            reranked_matches = res['output']
    else:
        reranked_matches = res['matches']

    COMPLETIONS_MODEL = llm
    answer_list = []
    if answer_per_paper:
        reranked_matches_grouped = concatenate_chunks_of_same_paper(
            reranked_matches)

        for match in reranked_matches_grouped:
            prompt_citation = construct_prompt_per_paper(
                query,
                template,
                COMPLETIONS_MODEL,
                match)

            if prompt_citation['code'] == 'failure':
                logger.error(prompt_citation['output'])
                answer_list.append({'context': match['metadata']['text'],
                                    'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                    'reference': []})
            else:
                prompt = prompt_citation['prompt']
                print(
                    f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

                # Log prompt
                # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

                # pdb.set_trace()

                # Call chat completion API
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": temp,
                    # "temperature": 0.9,
                    # "max_tokens": 1000,
                    "max_tokens": 1000 if COMPLETIONS_MODEL == "gpt-4" else 600,
                    "model": COMPLETIONS_MODEL
                }

                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **COMPLETIONS_API_PARAMS
                    )
                    answer = response["choices"][0]["message"]["content"].strip(
                        " \n")
                    print(
                        f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                    answer_list.append({'context': match['metadata']['text'],
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

                except Exception as e:
                    msg = f"Response error from OpenaAI! Actual exception is: {e}"
                    logger.error(msg)
                    answer_list.append({'context': match['metadata']['text'],
                                        'answer': msg,
                                        'reference': []})

    else:
        reranked_matches = order_chunks_of_same_paper(reranked_matches)

        prompt_citation = construct_prompt(
            query,
            template,
            COMPLETIONS_MODEL,
            reranked_matches)
        if prompt_citation['code'] == 'failure':
            logger.error(prompt_citation['output'])
            answer_list.append({'context': prompt_citation['context'],
                                'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                'reference': []})
        else:
            prompt = prompt_citation['prompt']
            print(
                f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")
            # citation_query = prompt_citation['citation']

            # Log prompt
            # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

            # completion api
            COMPLETIONS_API_PARAMS = {
                # We use temperature of 0.0 because it gives the most predictable, factual answer.
                "temperature": temp,
                # "temperature": 0.9,
                # "max_tokens": 1000,
                "max_tokens": 1200 if COMPLETIONS_MODEL == "gpt-4" else 800,
                "model": COMPLETIONS_MODEL
            }

            try:
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    **COMPLETIONS_API_PARAMS
                )
                answer = response["choices"][0]["message"]["content"].strip(
                    " \n")
                print(
                    f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': answer,
                                    'reference': prompt_citation['citation']})

            except Exception as e:
                msg = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(msg)
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': msg,
                                    'reference': []})
    # pdb.set_trace()
    return {'code': 'success', 'output': answer_list}


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
# def comma_sep_strings2list(search_keywords):
#     search_keywords = search_keywords.split(",")
#     search_keywords = [s for s in search_keywords if s]
#     return search_keywords

# def get_keywords_from_query(query):
#     # Find all instances enclosed with double asterisks
#     double_star_matches = re.findall(r"\*\*(.*?)\*\*", query)
    
#     # Find all instances enclosed with single or double asterisks
#     single_star_matches = re.findall(r"(\*{1,2})(.*?)(\1)", query)
    
#     # Extract only the matched content from single_star_matches, ignoring the asterisks
#     single_star_matches = [match[1] for match in single_star_matches]
    
#     # Replace single and double asterisks with an empty string
#     query_without_stars = re.sub(r"\*+", "", query)
    
#     return {"filtered_query":query_without_stars, "search_keywords": single_star_matches, "primary_keywords":double_star_matches}
###############################################################################



###############################################################################

def scientific_qa_bot_GUI_rerank(namespace,
                                 search_keywords,
                                 primary_keywords,
                                 paper_id=-1,
                                 chunks_from_one_paper=False,
                                 keyword_fixed=False,
                                 rerank_flag=False):
           
    ## Handle the exception when search_keywords and primary_keywords are empty
    if keyword_fixed and not bool(primary_keywords):
        return {'code': 'failure', 'msg': f'keyword_fixed is {bool(keyword_fixed)} but primary_keywords list is empty'}
    elif rerank_flag and not bool(search_keywords):
        return {'code': 'failure', 'msg': f'rerank_flag is {bool(rerank_flag)} but search_keywords list is empty'}

    # index = pmcd.get_pinecone_index()
    
    ## Load all vectors in a namespace
    start = pmcd.time.time()  # time modiule was imported in pmcd, so use from there
    # res = {"matches": load_all_chunks_in_namespace(namespace)}
    load_res = load_all_chunks_in_namespace(namespace)
    end = pmcd.time.time()
            
    if load_res['code'] == 'failure':
        return {"code": load_res['code'], "msg": load_res['msg']}
    else:
        res = {"matches": load_res['chunks_in_namespace']}        
        # logger.info(f"Time taken to load {len(res['matches'])} chunks = {round(end - start, 3)} seconds")
        # print(f"Time taken to load {len(res['matches'])} chunks = {round(end - start, 3)} seconds")
    
    ## Check if namespace is non-empty
    if not res["matches"]:
        output = "The dict key 'matches' has zero values, did you select correct namespace?"
        logger.error(output)
        return {"code": "failure", "msg": output}
    
    # Extract chunks from a single paper
    # Here also handle the exception for invalid paper_id, becaus ethat will generate empty res!
    if chunks_from_one_paper:
        res = extract_chunks_from_one_paper(res, paper_id)

    # Re-rank pinecone chunks based on keyword frequency
    ranked_df = rerank_matches_GUI(res, search_keywords, keyword_fixed, primary_keywords)
    # top_n_rows = 50 if ranked_df.shape[0] >= 50 else ranked_df.shape[0]
    # return {"code": 'success', "output": ranked_df[:top_n_rows]}
    
    ## return all chunks that contain where the primary_keywords atleast appear
    df_final = ranked_df[(ranked_df[primary_keywords] !=0).all(axis=1)]
    # pdb.set_trace()


    if df_final.empty:
        return {"code": 'failure', "msg": "Empty keyword frequency table! check for typo in your keyword(s) and also make sure you are searching in revelant namespace"}
    else:
        return {"code": 'success', "output": df_final}
###############################################################################


def return_answer_GUI_rerank(
                             vector_ids, 
                             llm,
                             temp,
                             namespace, 
                             query, 
                             search_keywords,
                             template,
                             paper_id,
                             answer_per_paper, 
                             chunks_from_one_paper
                             ):    
    
    COMPLETIONS_MODEL = llm
    
    ## Log query
    query_logger.info(f"Q:- {query}")

    ## Load all chunks in a namespace
    load_res = load_all_chunks_in_namespace(namespace)
    if load_res['code'] == 'failure':
        return {"code": load_res['code'], "msg": load_res['msg']}
    else:
        res = {"matches": load_res['chunks_in_namespace']} 
        
    ## Check if namespace is non-empty
    if not res["matches"]:
        output = "The dict key 'matches' has zero values, did you select correct namespace?"
        logger.error(output)
        return {"code": "failure", "msg": output}
    
    # pdb.set_trace()
    
    ## Write logic to extract chunks using vector_ids
    vector_ids_set = set(vector_ids)  # Convert list to set for efficient lookups
    filtered_chunks = [d for d in res["matches"] if d['id'] in vector_ids_set]
    
    # Reorder the filtered chunks based on the order in lk_list
    filtered_chunks.sort(key=lambda x: vector_ids.index(x['id']))
    
    # update res with filtered chunks only
    # res['matches'] = copy.deepcopy(filtered_chunks)

    res['matches'] = filtered_chunks
    
    ## Then based on some flag concatenate_chunks_of_same_paper
    answer_list = []
    if answer_per_paper:
        reranked_matches_grouped = concatenate_chunks_of_same_paper(res['matches'])        
        for match in reranked_matches_grouped:
                prompt_citation = construct_prompt_per_paper(
                query,
                template,
                COMPLETIONS_MODEL,
                match,
                search_keywords)
                if prompt_citation['code'] == 'failure':
                    logger.error(prompt_citation['output'])
                    answer_list.append({'context': match['metadata']['text'],
                                        'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                        'reference': []})
                else:
                    prompt = prompt_citation['prompt']
                # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

                # Log prompt
                # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

                # pdb.set_trace()

                # Call chat completion API
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": temp,
                    "max_tokens": 1000 if COMPLETIONS_MODEL == "gpt-4" else 600,
                    "model": COMPLETIONS_MODEL
                }

                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **COMPLETIONS_API_PARAMS
                    )
                    answer = response["choices"][0]["message"]["content"].strip(" \n")
                    # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                    
                    # pdb.set_trace()
                    answer_list.append({'context': mark_keywords(match['metadata']['text'], search_keywords),
                                        # 'context': match['metadata']['text'],
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

                except Exception as e:
                    msg = f"Response error from OpenaAI! Actual exception is: {e}"
                    logger.error(msg)
                    answer_list.append({'context': match['metadata']['text'],
                                        'answer': msg,
                                        'reference': []})
                    
    ## logic for both chunks_from_one_paper = True
    elif chunks_from_one_paper:
        reranked_matches = extract_chunks_from_one_paper(res, paper_id)
        if not reranked_matches['matches']:
            output = f"The dict key 'matches' has zero values! check paper_id. Provided paper_id = {paper_id}"
            logger.error(output)
            return {"code": "failure", "msg": output}
        else:    
            prompt_citation = construct_prompt(
                query,
                template,
                COMPLETIONS_MODEL,
                reranked_matches,
                search_keywords)
            if prompt_citation['code'] == 'failure':
                logger.error(prompt_citation['output'])
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                    'reference': []})
            else:
                prompt = prompt_citation['prompt']
                # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

                # Log prompt
                # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

                # completion api
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": temp,
                    "max_tokens": 1200 if COMPLETIONS_MODEL == "gpt-4" else 800,
                    "model": COMPLETIONS_MODEL
                }
                
                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **COMPLETIONS_API_PARAMS
                    )
                    answer = response["choices"][0]["message"]["content"].strip(" \n")
                    # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                    
                    # pdb.set_trace()
                    answer_list.append({'context': prompt_citation['context'],
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

                except Exception as e:
                    msg = f"Response error from OpenaAI! Actual exception is: {e}"
                    logger.error(msg)
                    answer_list.append({'context': prompt_citation['context'],
                                        'answer': msg,
                                        'reference': []})                         
    else:
        reranked_matches = {'matches': order_chunks_of_same_paper(res['matches'])}
        prompt_citation = construct_prompt(
            query,
            template,
            COMPLETIONS_MODEL,
            reranked_matches,
            search_keywords)
        if prompt_citation['code'] == 'failure':
            logger.error(prompt_citation['output'])
            answer_list.append({'context': prompt_citation['context'],
                                'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                'reference': []})
        else:
            prompt = prompt_citation['prompt']
            # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

            # Log prompt
            # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

            # completion api
            model_lengths = {
            "gpt-4": 1200,
            "gpt-4-1106-preview": 2400,  # Replace with actual value
            "gpt-3.5-turbo-1106": 1200  # Replace with actual value
            }
            # Set a default length if the model is not in the dictionary
            default_length = 800
                    
            COMPLETIONS_API_PARAMS = {
                # We use temperature of 0.0 because it gives the most predictable, factual answer.
                "temperature": temp,
                # "max_tokens": 1200 if COMPLETIONS_MODEL == "gpt-4" else 800,
                "max_tokens": model_lengths.get(COMPLETIONS_MODEL, default_length),
                "model": COMPLETIONS_MODEL
            }

            try:
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    **COMPLETIONS_API_PARAMS
                )
                answer = response["choices"][0]["message"]["content"].strip(" \n")
                # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                
                # pdb.set_trace()
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': answer,
                                    'reference': prompt_citation['citation']})

            except Exception as e:
                msg = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(msg)
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': msg,
                                    'reference': []})    
    return {'code': 'success', 'output': answer_list}    



    
    ## call 
    
    # pass


###############################################################################


def scientific_qa_bot_GUI(llm,
                          temp,
                          namespace,
                          query,
                          template,
                          search_keywords,
                        #   primary_keywords,
                          embedd_model,
                          paper_id=-1,
                          answer_per_paper=False,
                          chunks_from_one_paper=False,
                        #   keyword_fixed=False,
                        #   rerank_flag=False,
                          top_k=5):
    
    ## Log query
    query_logger.info(f"Q:- {query}")

    # search_keywords = get_list_of_keywords() ## uncomment this line when running in production

    index = pmcd.get_pinecone_index()

    # ## compute OpenAI embedding
    # xq = openai.Embedding.create(input=query, engine=EMBEDDING_MODEL)['data'][0]['embedding']

    # compute BioBERT embedding (Do not need now, because all vectors are being used)
    xq = pmcd.get_text_embeddings([query], embedd_model, openai_vec_len=1536)

    # Query vector Db for chunks that matches with query embeeding
    start = pmcd.time.time()  # time modiule was imported in pmcd, so use from there
    if embedd_model == 'openai':
        res = index.query(vector=[xq], top_k=top_k, include_metadata=True, namespace=namespace) ## make xq a list of list before calling
    elif embedd_model == 'biobert':
        res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace) ## xq is already list of list

    # # xq is already list of list, required for pincone query function
    # res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)

    end = pmcd.time.time()
    # logger.info(f"Time taken to fetch {len(res['matches'])} matching chunks = {round(end - start, 3)} seconds")

    # res["matches"] = remove_values_key_pinecone_res(res["matches"])
    # pdb.set_trace()

    if not res["matches"]:
        output = "The dict key 'matches' has zero values, did you select correct namespace?"
        logger.error(output)
        return {"code": "failure", "msg": output}

    # Extract chunks from a single paper
    if chunks_from_one_paper:
        res = extract_chunks_from_one_paper(res, paper_id)
        answer_per_paper = False
        # pdb.set_trace()

    # # Re-rank pinecone chunks based on keyword frequency
    # # rerank_flag = input("Enter 1 if you want to re-rank chunks before prompt generation, otherewise press any key = ")
    # if rerank_flag:
    #     res = rerank_matches(res, search_keywords,
    #                          keyword_fixed, primary_keywords)
    #     if res['code'] == 'failure':
    #         logger.error(res['output'])
    #         return {"code": res['code'], "msg": res['output']}
    #     else:
    #         reranked_matches = res['output']
    # else:
    #     reranked_matches = res['matches']
    
    # reranked_matches = res['matches']
    COMPLETIONS_MODEL = llm
    answer_list = []
    if answer_per_paper:
        # reranked_matches_grouped = concatenate_chunks_of_same_paper(reranked_matches)
        res['matches'] = concatenate_chunks_of_same_paper(res['matches'])

        # for match in reranked_matches_grouped:
        for match in res['matches']:
            prompt_citation = construct_prompt_per_paper(
                query,
                template,
                COMPLETIONS_MODEL,
                match,
                search_keywords)

            if prompt_citation['code'] == 'failure':
                logger.error(prompt_citation['output'])
                answer_list.append({'context': match['metadata']['text'],
                                    'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                    'reference': []})
            else:
                prompt = prompt_citation['prompt']
                # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

                # Log prompt
                # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

                # pdb.set_trace()

                # Call chat completion API
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": temp,
                    # "temperature": 0.9,
                    # "max_tokens": 1000,
                    "max_tokens": 1000 if COMPLETIONS_MODEL == "gpt-4" else 600,
                    "model": COMPLETIONS_MODEL
                }

                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **COMPLETIONS_API_PARAMS
                    )
                    answer = response["choices"][0]["message"]["content"].strip(
                        " \n")
                    # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                    # answer_list.append({'context': match['metadata']['text'],
                    #                     'answer': answer,
                    #                     'reference': prompt_citation['citation']})
                    
                    answer_list.append({'context': mark_keywords(match['metadata']['text'], search_keywords),
                                        # 'context': match['metadata']['text'],
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

                except Exception as e:
                    msg = f"Response error from OpenaAI! Actual exception is: {e}"
                    logger.error(msg)
                    answer_list.append({'context': match['metadata']['text'],
                                        'answer': msg,
                                        'reference': []})
    
    ## logic for both chunks_from_one_paper = True
    elif chunks_from_one_paper:
        # reranked_matches = extract_chunks_from_one_paper(res, paper_id)
        res = extract_chunks_from_one_paper(res, paper_id)
        # if not reranked_matches['matches']:
        if not res['matches']:
            output = f"The dict key 'matches' has zero values! check paper_id. Provided paper_id = {paper_id}"
            logger.error(output)
            return {"code": "failure", "msg": output}
        else:    
            prompt_citation = construct_prompt(
                query,
                template,
                COMPLETIONS_MODEL,
                # reranked_matches,
                res,
                search_keywords)
            if prompt_citation['code'] == 'failure':
                logger.error(prompt_citation['output'])
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                    'reference': []})
            else:
                prompt = prompt_citation['prompt']
                # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")

                # Log prompt
                # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

                # completion api
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": temp,
                    "max_tokens": 1200 if COMPLETIONS_MODEL == "gpt-4" else 800,
                    "model": COMPLETIONS_MODEL
                }
                
                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **COMPLETIONS_API_PARAMS
                    )
                    answer = response["choices"][0]["message"]["content"].strip(" \n")
                    # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                    
                    # # pdb.set_trace()
                    # answer_list.append({'context': prompt_citation['context'],
                    #                     'answer': answer,
                    #                     'reference': prompt_citation['citation']})
                    
                    answer_list.append({'context': mark_keywords(prompt_citation['context'], search_keywords),
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

                except Exception as e:
                    msg = f"Response error from OpenaAI! Actual exception is: {e}"
                    logger.error(msg)
                    answer_list.append({'context': prompt_citation['context'],
                                        'answer': msg,
                                        'reference': []})  

    else:
        # reranked_matches = order_chunks_of_same_paper(reranked_matches)
        
        # pdb.set_trace()
        res['matches'] = order_chunks_of_same_paper(res['matches'])

        prompt_citation = construct_prompt(
            query,
            template,
            COMPLETIONS_MODEL,
            # reranked_matches,
            res,
            search_keywords)
        # pdb.set_trace()
        if prompt_citation['code'] == 'failure':
            logger.error(prompt_citation['output'])
            answer_list.append({'context': prompt_citation['context'],
                                'answer': prompt_citation['code'] + "! " + prompt_citation['output'],
                                'reference': []})
        else:
            prompt = prompt_citation['prompt']
            # logger.info(f"Total number of tokens in prompt = {num_tokens_from_string(prompt, COMPLETIONS_MODEL)}\n")
            # citation_query = prompt_citation['citation']

            # Log prompt
            # prompt_logger.info(prompt + "\n>>>> END PROMPT >>>>\n\n")

            # completion api
            COMPLETIONS_API_PARAMS = {
                # We use temperature of 0.0 because it gives the most predictable, factual answer.
                "temperature": temp,
                # "temperature": 0.9,
                # "max_tokens": 1000,
                "max_tokens": 1200 if COMPLETIONS_MODEL == "gpt-4" else 800,
                "model": COMPLETIONS_MODEL
            }

            try:
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    **COMPLETIONS_API_PARAMS
                )
                answer = response["choices"][0]["message"]["content"].strip(" \n")
                # logger.info(f"Total number of tokens in completion = {num_tokens_from_string(answer, COMPLETIONS_MODEL)}\n")
                # answer_list.append({'context': prompt_citation['context'],
                #                     'answer': answer,
                #                     'reference': prompt_citation['citation']})
                answer_list.append({'context': mark_keywords(prompt_citation['context'], search_keywords),
                                        'answer': answer,
                                        'reference': prompt_citation['citation']})

            except Exception as e:
                msg = f"Response error from OpenaAI! Actual exception is: {e}"
                logger.error(msg)
                answer_list.append({'context': prompt_citation['context'],
                                    'answer': msg,
                                    'reference': []})
    return {'code': 'success', 'output': answer_list}
###############################################################################


def get_embeddings(query, top_k, namespace):
    index = pmcd.get_pinecone_index()
    xq = openai.Embedding.create(input=query, engine=EMBEDDING_MODEL)[
        'data'][0]['embedding']

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
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        answer = response["choices"][0]["message"]["content"].strip(" \n")
        return {"code": "success", "msg": answer}
    except Exception as e:
        msg = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
        return {"code": "error", "msg": msg}
###############################################################################


# def entity_extraction(text, prompt_init, llm):
#     # prompt = "Summarize this for a second-grade student:\n\n" + text
#     prompt = prompt_init + text
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     try:
#         response = openai.ChatCompletion.create(
#             model=llm,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=1,
#             max_tokens=500,
#             top_p=1.0,
#             frequency_penalty=0.0,
#             presence_penalty=0.0
#         )
#         answer = response["choices"][0]["message"]["content"].strip(" \n")
#         return {"code": "success", "msg": answer}
#     except Exception as e:
#         msg = f"Error running openai.ChatCompletion.create(...)! Actual exception is: {e}"
#         return {"code": "error", "msg": msg}
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
