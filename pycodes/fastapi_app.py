from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
import pickle
import json
import asyncio
import time

## import utilities  
import pycodes.PMC_downloader_Utils as pmcd
import pycodes.QA_OpenAI_api_Utils as qa

# Setup logger
# log_dir = os.getcwd() + "/Logs"
# log_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/Logs"
# log_dir = os.path.abspath("../Logs") 
log_dir = os.path.abspath("./Logs") 
log_file_fast = "fastAPI.log"

# Create a global variable to log across different places in this file
logger = pmcd.create_logger(log_dir, log_file_fast, 'weiseeule_logger')

# create app object
app = FastAPI()

# check app status
@app.get("/health")
def read_health():
    return {"status": "OK"}

# An example of a request model
class SomeParameterModel(BaseModel):
    param1: str
    param2: int
    # Add other fields as required

# User validation model
class User(BaseModel):
    username: str
    password: str

# Search PMID in Namespace model
class SearchNamespace(BaseModel):
    namespace: str
    pmid: str

# Fetch Articles
class fetchArticles(BaseModel):    
    embedd_model: str
    keywords: str
    start_date: str
    end_date: str

# Summarize abstracts for selected PMIDs
class SummarizeAbstracts(BaseModel):
    llm: str
    pmids: str
    query: str

class GetAnswer(BaseModel):
    llm: str
    # temp: str
    temp: float
    namespace: str
    query: str
    template: str
    embedd_model: str
    paper_id: str
    answer_per_paper: str
    chunks_from_one_paper: str
    fix_keyword: str
    rerank: str
    top_k: str    

class SearchPubMed(BaseModel):
    query: str

##------------------------------------------------------------------##
# Endpoint for searching a pmid inside namespace
@app.websocket("/ws/search_PMID_in_namespace")
async def websocket_search_PMID_in_namespace(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            params = await websocket.receive_json()
            params = SearchNamespace(**params)
            logger.info("Inside search_PMID_in_namespace endpoint()")
            logger.info(f"params.namespace = {params.namespace}")
            logger.info(f"params.pmid = {params.pmid}")
            async for message in pmcd.check_article_contents(params.namespace, params.pmid):
                await websocket.send_text(json.dumps(message))
    except Exception as e:
            logger.error(f"Error in summarize abstracts WebSocket endpoint: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))

##------------------------------------------------------------------##
## Websocket endpoint for testing
@app.websocket("/ws/test_socket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
        except WebSocketDisconnect:
            break
##------------------------------------------------------------------##

## Websocket endpoint for streaming the response
@app.websocket("/ws/stream_answer")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Inside websocket_endpoint()")
    await websocket.accept()    
    while True:
        logger.info("Inside while True:")
        try:
            logger.info("Before await websocket.receive_json()")
            # params = await websocket.receive_json()
            try:
                params = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout error: Did not receive data within 10 seconds")
                await websocket.send_text(json.dumps({"error": "Timeout: Did not receive data within 10 seconds"}))
                break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break

            logger.info(f"Received params: {params}")
            params = GetAnswer(**params)
            logger.info(f"Parsed params: {params}")
            rerank_flag = pmcd.string2bool(params.rerank)
            query = params.query
            if rerank_flag:
                logger.info("Inside rerank_flag block")
                keyword_query = qa.get_keywords_from_query(query.lower(), params.llm)
                logger.info(f"keyword_query: {keyword_query}")
                if keyword_query['code'] == 'failure':
                    result = {"error": keyword_query["msg"] + "Try using gpt-4 series LLM."}
                    # return result
                    await websocket.send_text(json.dumps(result))
                with open('auto_extracted_keywords.pkl', 'wb') as f:
                    pickle.dump(keyword_query, f)
                query = keyword_query['filtered_query']
                search_keywords = keyword_query['search_keywords']
                primary_keywords = keyword_query['primary_keywords']
            else:
                search_keywords = []
                primary_keywords = []

            logger.info(f"Inside fastAPI app, before calling qa.scientific_qa_bot_GUI_stream() query = {params.query}")   
            
            async for message in qa.scientific_qa_bot_GUI_stream(params.llm, 
                                                            # float(params.temp), 
                                                            params.temp,
                                                            params.namespace,
                                                            params.query, 
                                                            params.template, 
                                                            search_keywords,
                                                            primary_keywords,
                                                            params.embedd_model,
                                                            int(params.paper_id),
                                                            pmcd.string2bool(params.fix_keyword),                                                            
                                                            pmcd.string2bool(params.answer_per_paper),
                                                            pmcd.string2bool(params.chunks_from_one_paper),
                                                            rerank_flag,
                                                            int(params.top_k)):
                await websocket.send_text(json.dumps(message))
                # logger.info(f"After await websocket.send_text(...)")
        except Exception as e:
            logger.error(f"Error in WebSocket endpoint: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))
##------------------------------------------------------------------##

## Websocket endpoint for searching PubMed
@app.websocket("/ws/search_pubmed")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Inside search pubmed websocket endpoint")
    await websocket.accept() 
    while True:
        try:
            try:
                params = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout error: Did not receive data within 10 seconds")
                await websocket.send_text(json.dumps({"error": "Timeout: Did not receive data within 10 seconds"}))
                break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            logger.info(f"Received params: {params}")
            params = SearchPubMed(**params)
            logger.info(f"Parsed params: {params}")
            async for message in qa.search_PubMed(params.query):
                await websocket.send_text(json.dumps(message))
                    # message = 'Hello from search_pubmed websocket endpoint'
        except Exception as e:
            logger.error(f"Error in search pubmed WebSocket endpoint: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))
##------------------------------------------------------------------##

@app.websocket("/ws/summarize_abstracts")
async def summarize_abstracts_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            try:
                params = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout error: Did not receive data within 10 seconds")
                await websocket.send_text(json.dumps({"error": "Timeout: Did not receive data within 10 seconds"}))
                break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            logger.info(f"Insdie summarize_abstracts and received params: {params}")
            params = SummarizeAbstracts(**params)
            pmids = params.pmids # creating list from by splitting at commas
            pmids = pmids.split(',')
            logger.info(f"Parsed params: {params}")
            logger.info(f"type(pmids) = {type(pmids)}")
            if pmids:
                async for message in qa.process_pmids_and_summarize(pmids, params.llm, params.query):
                    await websocket.send_text(json.dumps(message))
            else:
                await websocket.send_text(json.dumps({"error": "No PMIDs provided"}))
        except Exception as e:
            logger.error(f"Error in summarize abstracts WebSocket endpoint: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))

##------------------------------------------------------------------##

@app.websocket("/ws/fetch_articles")
async def fetch_articles_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            params = await websocket.receive_json()
            params = fetchArticles(**params)
            logger.info("Inside fetch_articles endpoint()")
            logger.info(f"params = {params}")
            
            ## Get the arguments
            start = time.time()
            embedd_model = params.embedd_model
            keywords = params.keywords # keyword which should appear in title/abstact
            start_date = params.start_date
            end_date = params.end_date

            ## Check if the local database exists; if not, create it
            # local_db_name = ('_'.join(keywords.split()) + "_local.db").lower()
            namespace = ('_'.join(keywords.lower().split()) + '_' + embedd_model)
            local_DB_dir = "Local_DB"
            os.makedirs(local_DB_dir, exist_ok=True)
            # local_db_path = "./" + local_DB_dir + "/" + local_db_name
            local_db_path = './' + local_DB_dir + '/' + namespace + '.db'

            if not os.path.isfile(local_db_path):
                res = pmcd.create_tables(local_db_path)
                if res["code"] == "failure":  
                    await websocket.send_text(json.dumps({"code": "failure", "message": res["msg"]}))
                else:
                    ## Insert user into local DB
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
                await websocket.send_text(json.dumps({"code": "exit", "message": res["msg"]}))

            # # temporary else
            # else:
            #     await websocket.send_text(json.dumps({"code": "success", "message": res["msg"]}))

            else:
                chunk_size = 2000
                # namespace = "_".join(keywords.split()) + "_c" + str(chunk_size)
                table_name = "Results"
                res = pmcd.preprocess_data_qa(local_db_path, 
                                        table_name, 
                                        namespace,
                                        embedd_model, 
                                        chunk_size, 
                                        data_dir="tmp")
                
                if res["code"] == "failure":
                    await websocket.send_text(json.dumps({"code": "failure", "message": res["msg"]}))
                else:
                    end = time.time()
                    msg = "Time taken to fetch articles and insert into local DB"
                    pmcd.log_elapsed_time(start, end, msg)
                    await websocket.send_text(json.dumps({"code": "success", 
                                                          "message": "Articles fetched, " + "Converted into vectors, " + res["msg"]}))
    except Exception as e:
            logger.error(f"Error in fetch articles WebSocket endpoint: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))
