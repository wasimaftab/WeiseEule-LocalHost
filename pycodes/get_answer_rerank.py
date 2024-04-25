import argparse
import json

# import time
import os
import pickle

# import PMC_downloader_Utils as pmcd
import QA_OpenAI_api_Utils as qa

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


def get_answer_rerank(args):
    reqd_params = 9
    # Get the arguments from JS
    num_params = len(args)
    if num_params != reqd_params:
        result = {
            "greeting": "Hello from get_answer_rerank.py!",
            "code": "failure",
            "output": f"return_answer_GUI_rerank(...) needs {reqd_params} params, but supplied {num_params}",
        }
    else:
        vector_ids = args.get('vector_ids', []) # Extract vector_ids, default to empty list if not present
        llm = args.get("llm")
        temp = float(args.get("temp"))
        namespace = args.get("namespace")
        # keyword_query = qa.get_keywords_from_query(args.get("query"))
        with open('auto_extracted_keywords.pkl', 'rb') as f:
            keyword_query = pickle.load(f)
        query = keyword_query['filtered_query']  
        search_keywords = keyword_query['search_keywords']
        template = args.get("template")
        paper_id = int(args.get("paper_id"))
        answer_per_paper = string2bool(args.get("answer_per_paper"))
        chunks_from_one_paper = string2bool(args.get("chunks_from_one_paper"))

        res = qa.return_answer_GUI_rerank(
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
            )
        if res["code"] == "success":
            result = {
                "greeting": "Hello from get_answer_rerank.py!",
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
                "greeting": "Hello from get_answer_rerank.py!",
                "llm": llm,
                "temp": temp,
                "namespace": namespace,
                "query": query,
                "pwd": os.getcwd(),
                "code": res["code"],
                "output": res["msg"],
            }
    print(json.dumps(result))


# Set up argument parser object and fetch the args
parser = argparse.ArgumentParser()
parser.add_argument("json_args")
args = parser.parse_args()
json_args = json.loads(args.json_args)
get_answer_rerank(json_args)
