import argparse
import json

# import time
import os

import PMC_downloader_Utils as pmcd
import QA_OpenAI_api_Utils as qa
# import pdb
import re
import pickle


def string2bool(val):
    if val == "True":
        val = True
    else:
        val = False
    return val


def string2list(search_keywords):
    search_keywords = search_keywords.split(",")
    search_keywords = [s for s in search_keywords if s]
    return search_keywords


def get_answer(args):
    reqd_params = 12 
    # Get the arguments from JS
    num_params = len(args)
    if num_params != reqd_params:
        result = {
            "greeting": "Hello from get_answer.py!",
            "code": "failure",
            "message": f"get_answer() needs {reqd_params} params, but supplied {num_params}",
        }
        print(json.dumps(result))
    else:        
        llm = args.get("llm")
        temp = float(args.get("temp"))
        namespace = args.get("namespace")
        # print(f"Inside get_answer.py and query = {args.get('query')}")
        keyword_query = qa.get_keywords_from_query(args.get("query").lower(), llm)
        if keyword_query['code'] == 'failure':
            result = {"code": keyword_query["code"],                     
                      "output": keyword_query["msg"]}
            print(json.dumps(result))
        with open('auto_extracted_keywords.pkl', 'wb') as f:
            pickle.dump(keyword_query, f)
        query = keyword_query['filtered_query']        
        template = args.get("template")
        search_keywords = keyword_query['search_keywords']
        primary_keywords = keyword_query['primary_keywords']
        embedd_model = args.get("embedd_model")
        paper_id = int(args.get("paper_id"))
        answer_per_paper = string2bool(args.get("answer_per_paper"))
        chunks_from_one_paper = string2bool(args.get("chunks_from_one_paper"))
        keyword_fixed = string2bool(args.get("keyword_fixed"))
        rerank_flag = string2bool(args.get("rerank"))
        top_k = int(args.get("top_k"))
        

        if rerank_flag:
            pmcd.logger.info("Calling qa.scientific_qa_bot_GUI_rerank()")
            res = qa.scientific_qa_bot_GUI_rerank(
                namespace,
                search_keywords,
                primary_keywords,
                paper_id,
                chunks_from_one_paper,
                keyword_fixed,
                rerank_flag
            )
            pmcd.logger.info(f"res['code'] = {res['code']}")
            if res["code"] == "success":
                result = {
                    "code": res["code"],
                    "pwd": os.getcwd(),
                    "output": res["output"].to_dict(orient='records')}
            else:
                result = {
                    "code": res["code"],                     
                    "pwd": os.getcwd(),
                    "output": res["msg"]}
            print(json.dumps(result))
            # print(json.dumps(res["output"].to_dict(orient='records')))

        else:
            res = qa.scientific_qa_bot_GUI(
                llm,
                temp,
                namespace,
                query,
                template,
                search_keywords,
                # primary_keywords,
                embedd_model,
                paper_id,
                answer_per_paper,
                chunks_from_one_paper,
                # keyword_fixed,
                # rerank_flag,
                top_k)
            
            # NEW version of getting result
            if res["code"] == "success":
                result = {
                    "greeting": "Hello from get_answer.py!",
                    "llm": llm,
                    "temp": temp,
                    "namespace": namespace,
                    "query": query,
                    "pwd": os.getcwd(),
                    "code": res["code"],
                    "output": res["output"],
                }
            else:
                result = {
                    "greeting": "Hello from get_answer.py!",
                    "llm": llm,
                    "temp": temp,
                    "namespace": namespace,
                    "query": query,
                    "pwd": os.getcwd(),
                    "code": res["code"],
                    "output": res["msg"],
                }
            print(json.dumps(result))

    # Call this function at the end of function 
    # pmcd.record_memory_usage()


# Set up argument parser object and fetch the args
parser = argparse.ArgumentParser()
parser.add_argument("json_args")
args = parser.parse_args()
json_args = json.loads(args.json_args)
get_answer(json_args)

