#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:26:29 2024

@author: wasim

Extract all meta data from a namespace in pinecone DB
"""

import numpy as np
import QA_OpenAI_api_Utils as qa
import PMC_downloader_Utils as pmcd
import copy
import time
import pickle
import sys
import pdb

## Select a namespace
namespace = input("Enter a namespace name, must match with the one in pinecone server = ")

index = pmcd.get_pinecone_index()

def get_matches_from_query(index, input_vector, top_k, namespace):
    print("\nsearching pinecone...")
    results = index.query(vector=input_vector, 
                          top_k=top_k, 
                          include_metadata=True, 
                          namespace=namespace, 
                          include_values=False)
    ids = set()
    print(type(results))
    for result in results['matches']:
        ids.add(result['id'])
    return {"matches": results['matches'], "ids": ids}


def get_all_matches_from_index(index, num_dimensions, namespace=""):   
    # Get number of vectors in the namespace
    index_stats = index.describe_index_stats()
    num_vectors = index_stats.get('namespaces')[namespace].vector_count    
    all_matches = []
    all_ids = set()
    prev_ids = set()
    ids = set()
    prev_matches = set()
    loop = 0
    top_k = 7000
    while len(all_ids) < num_vectors:
        print("Length of ids list is shorter than the number of total vectors...")
        input_vector = np.random.rand(num_dimensions).tolist()
        print("creating random vector...")
        res = get_matches_from_query(index, input_vector, top_k, namespace)
        all_matches.append(res['matches'])
        ids = res['ids']

        prev_ids = copy.deepcopy(all_ids)
        all_ids.update(ids)
        if loop:
            delta_fetch = all_ids.difference(prev_ids)
            print(f"{len(delta_fetch)} new ids are fetched in loop {loop+1}")
        loop += 1
        print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
        time.sleep(60)
        
    ## convert list of lists to list, i.e. [[], [], ...] --> [...]
    records = []
    for alist in all_matches:
        for l in alist:
            records.append(l)
    return {"all_matches": records, "all_ids": all_ids}

index = pmcd.get_pinecone_index()

matches_ids_dict = get_all_matches_from_index(index, num_dimensions=1536, namespace=namespace)

## Now remove redundant rocords
seen_ids = set()
unique_matches = []

for d in matches_ids_dict['all_matches']:
    id_val = d['id']
    if id_val not in seen_ids:
        seen_ids.add(id_val)
        unique_matches.append(d)

## There are empty values field, first remove that from each dict
unique_matches_empty_removed = qa.remove_values_key_pinecone_res(unique_matches)

## Check indeed it has all the records
unique_ids = []
for match in unique_matches_empty_removed:
    unique_ids.append(match['id'])
    
if sorted(matches_ids_dict['all_ids']) == sorted(unique_ids):
    print('unique_matches_empty_removed contains all the unique records')
else:
    sys.exit('unique_matches_empty_removed does NOT contain all the unique records')

# pdb.set_trace()
## Now save the unique_matches
with open('../Pickled_vecs/' + namespace + '_chunks.pkl', 'wb') as f:
    pickle.dump(unique_matches_empty_removed, f)
    
# ## To load the unique_matches
# with open(namespace + '_chunks.pkl', 'rb') as f:
#     chunks_in_namespace = pickle.load(f)    
# https://docs.pinecone.io/docs/manage-data
# index.fetch(list(all_ids))
