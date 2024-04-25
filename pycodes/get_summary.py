import argparse
import json
# import time
import os
# import PMC_downloader_Utils as pmcd
import QA_OpenAI_api_Utils as qa

def get_summary(args):
    ## Get the arguments from JS
    llm = args.get('llm')
    # temp = args.get('temp')
    text = args.get('text')
    prompt_init = "Summarize this for a scientist:\n\n" 
    res = qa.summarize_text(text, prompt_init, llm)
    result = {
    "greeting": "Hello from get_summary.py!",
    "llm" : llm,
    "prompt_init" : prompt_init,
    "text": text,
    "code": res["code"],
    "message": res["msg"]
    }
    print(json.dumps(result))

## Set up argument parser object and fetch the args   
parser = argparse.ArgumentParser()
parser.add_argument('json_args')
args = parser.parse_args()
json_args = json.loads(args.json_args)
get_summary(json_args)