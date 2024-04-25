import argparse
import json
import os
import PMC_downloader_Utils as pmcd

def get_pmid_info(args):
    reqd_params = 2 
    # Get the arguments from JS
    num_params = len(args)
    if num_params != reqd_params:
        result = {
            "greeting": "Hello from get_pmid_info.py!",
            "code": "failure",
            "message": f"get_pmid_info() needs {reqd_params} params, but supplied {num_params}",
        }
        print(json.dumps(result))
    else:
        namespace = args.get("namespace")
        pmid = args.get("pmid")
        res = pmcd.check_article_contents(namespace, pmid)
        # if res["code"] == "success":
        #     result = {
        #             "code": res["code"],
        #             "pwd": os.getcwd(),
        #             "output": res["msg"]}
        # else:
        #     result = {
        #         "code": res["code"],                     
        #         "pwd": os.getcwd(),
        #         "output": res["msg"]}
        result = {
                "code": res["code"],
                "pwd": os.getcwd(),
                "output": res["msg"]
                }
        print(json.dumps(result))            

# Set up argument parser object and fetch the args
parser = argparse.ArgumentParser()
parser.add_argument("json_args")
args = parser.parse_args()
json_args = json.loads(args.json_args)
get_pmid_info(json_args)