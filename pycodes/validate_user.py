import argparse
import json
import os
import PMC_downloader_Utils as pmcd
# import QA_OpenAI_api_Utils as qa

# ## This function uses a DB to validate
# def validate_user(args):
#     ## Get the arguments from JS
#     username = args.get('username')
#     password = args.get('password')
#     res = pmcd.validate_user_password(username, password)
#     result = {
#     "greeting": "Hello from validate_user.py!",
#     "code": res["code"],
#     "message": res["msg"]
#     }
#     print(json.dumps(result))

## This function uses env valiables to validate
def validate_user(args):
    ## Get the arguments from JS
    username_in = args.get('username')
    password_in = args.get('password')

    ## Get the credentials from env vars
    username = os.getenv("WeiseEule_username")
    password = os.getenv("WeiseEule_password")

    ## validate and return
    if username_in == username and password_in == password:
        result = {"greeting": "Hello from validate_user.py!", "code": "success", "message": "Authentication successful"}
    else:
        result = {"greeting": "Hello from validate_user.py!", "code": "failure", "message": "Invalid username or password"}

    print(json.dumps(result))

## Set up argument parser object and fetch the args   
parser = argparse.ArgumentParser()
parser.add_argument('json_args')
args = parser.parse_args()
json_args = json.loads(args.json_args)
validate_user(json_args)