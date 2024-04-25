#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:54:34 2023

@author: wasim
"""
import argparse
import json
import time
import os
import PMC_downloader_Utils as pmcd
###############################################################################
def fetch_articles(args): 
    start = time.time()
    
    ## Get the arguments from JS
    embedd_model = args.get('embedd_model')
    keywords = args.get('keywords') # keyword which should appear in title/abstact
    start_date = args.get('start_date')
    end_date = args.get('end_date')

    ## Check if the local database exists; if not, create it
    # local_db_name = "qa_pmc_local.db" 
    local_db_name = ('_'.join(keywords.split()) + "_local.db").lower()
    local_DB_dir = "Local_DB"
    os.makedirs(local_DB_dir, exist_ok=True)
    local_db_path = "./" + local_DB_dir + "/" + local_db_name

    if not os.path.isfile(local_db_path):
        res = pmcd.create_tables(local_db_path)
        if res["code"] == "failure":  
            result = {
            "greeting": "Hello from fetch_artcles.py!",
            "keywords": keywords,
            "start_date": start_date,
            "end_date": end_date,
            "local_db_path": local_db_path,
            "pwd": os.getcwd(),
            "code": res["code"],
            "message": res["msg"]
            }
            print(json.dumps(result))
            return
        else:
            ## Insert user into local DB, for now having two dummy users so commented (ran alredy mannualy)
            users = [('testuser1', 'testpassword1'), ('testuser2', 'testpassword2')]
            pmcd.insert_users(local_db_path, users)
    
    ## Download papers on the given keyword and date range and push it into local DB
    res = pmcd.pmc_text_downloader(keyword=keywords,
                                  start_date=start_date,
                                  end_date=end_date,
                                  max_papers=10000,
                                  local_db_path=local_db_path,
                                  user_id=1) # hardcoded user_id 1 for now

    if res["code"] == "exit":
        result = {
        "greeting": "Hello from fetch_artcles.py!",
        "keywords": keywords,
        "start_date": start_date,
        "end_date": end_date,
        "local_db_path": local_db_path,
        "pwd": os.getcwd(),
        "code": res["code"],
        "message": res["msg"]
        }
        print(json.dumps(result))
        return
    else:
        chunk_size = 2000
        # namespace = "_".join(keywords.split())
        # namespace = "_".join(keywords.split())+"_c2000"
        # namespace = "_".join(keywords.split())+"_c" + chunk_size
        namespace = "_".join(keywords.split()) + "_c" + str(chunk_size)
        table_name = "Results"
        res = pmcd.preprocess_data_qa(local_db_path, 
                                table_name, 
                                namespace,
                                embedd_model, 
                                chunk_size, 
                                data_dir="tmp")

        if res["code"] == "failure":
            result = {
            "greeting": "Hello from fetch_artcles.py!",
            "embedd_model": embedd_model,
            "keywords": keywords,
            "start_date": start_date,
            "end_date": end_date,
            "local_db_path": local_db_path,
            "pwd": os.getcwd(),
            "code": res["code"],
            "message": res["msg"]
            }
        else:
            end = time.time()
            result = {
            "greeting": "Hello from fetch_artcles.py!",
            "keywords": keywords,
            "start_date": start_date,
            "end_date": end_date,
            "local_db_path": local_db_path,
            "pwd": os.getcwd(),
            "code": res["code"],
            "message": "Articles fetched, " + "Converted into vectors, " + res["msg"]
            }
            msg = "Time taken to fetch articles and insert into local DB"
            pmcd.log_elapsed_time(start, end , msg)

        print(json.dumps(result))

## Set up argument parser object and fetch the args   
parser = argparse.ArgumentParser()
parser.add_argument('json_args')
args = parser.parse_args()
json_args = json.loads(args.json_args)
fetch_articles(json_args)