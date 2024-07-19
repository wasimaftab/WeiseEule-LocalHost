#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions and variables needed to download full text article 
from PMC database. The functions so far allow the user to download full text, title, 
abstract, and citation for articles that have a keyword either in tile/abstract

Created on Wed Mar  24 14:52:25 2023

@author: wasim

TODO: In some cases the Mediline text_record do not contain abstacts, aim to extract
it from corresponding xml records
"""
from Bio import Entrez, Medline
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.error import HTTPError
from http.client import IncompleteRead
# from lxml import etree
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from tqdm import tqdm
# import openai
from openai import OpenAI
import pandas as pd
import os
import time
import re
# import sys
import logging
import sqlite3
# import pinecone
from pinecone import Pinecone
import shutil
import pdb
import xml.etree.ElementTree as ET
# from PyPDF2 import PdfReader
# import pickle
import torch
import numpy as np
import bcrypt
import psutil
import pickle
import glob

## Set model for word embedding
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_MODEL_BERT = "dmis-lab/biobert-base-cased-v1.2"
## Set your email address for Entrez API requests (required)
# email = "wasimgradapplication@gmail.com"
email=os.environ['EMAIL']

## Setup logger
# log_dir = "/home/wasim/Desktop/QA_Bot_Web_App/App/Logs"
log_dir = os.getcwd() + "/Logs"
log_file = "QA_bot.log"   
logger_name = 'weiseeule_logger'

## Set up openai client
syncClient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
###############################################################################
def string2bool(val):
    if val == "True":
        val = True
    else:
        val = False
    return val

def record_memory_usage():
    process = psutil.Process(os.getpid())
    memory_in_MB = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    with open("memory_log.txt", "a") as f:
        f.write(f"Memory used: {memory_in_MB} MB\n")


def create_logger(log_dir, log_file, logger_name):
    # Name your logger 
    logger = logging.getLogger(logger_name) 
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create log dir and file if does not exist
    log_path = os.path.join(log_dir, log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler that logs debug and higher level messages
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    # Specify the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)  
        
    return logger  

###############################################################################
## Create a global variable to log across different places in this file
logger = create_logger(log_dir, log_file, logger_name)
###############################################################################


def log_elapsed_time(start, end, msg):
    elapsed = end - start
    if 60 < elapsed < 3600:
        logger.info(f"{msg} = {round(elapsed/60, 3)} minutes")
    elif elapsed > 3600:
        logger.info(f"{msg} = {round(elapsed/3600, 3)} hours")
    else:
        logger.info(f"{msg} = {round(elapsed, 3)} seconds")
###############################################################################


def hash_password_bcrypt(password):
    # Hash a password for the first time, with a randomly-generated salt
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')

def check_password_bcrypt(hashed_password, user_password):
    # Check if a hashed password matches a user's password
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))

def validate_user_password(username, password):
    try:
        local_db_path = "Local_DB/authenticate_local.db"
        db_table = "Users"
        df = fetch_all_data_as_df(local_db_path, db_table)
        hashed_password =  df[df.username == username]['password'].iloc[0]
        if check_password_bcrypt(hashed_password, password):
            return {"code" : "success", "msg" : "You have access to WeiseEule"}
        else:
            return {"code" : "failure", "msg" : "You do not have access to WeiseEule"}
    except Exception as e:
        msg = f"Error validating user! Actual exception is: {e}"
        return {"code" : "error", "msg" : msg}




def create_authentication_table(local_db_name):
    try:
        with sqlite3.connect(local_db_name) as conn:
            c = conn.cursor()

            c.execute('''CREATE TABLE IF NOT EXISTS Users
                        (user_id INTEGER PRIMARY KEY, 
                        username TEXT, 
                        password TEXT)''')
            conn.commit()
        return {"code" : "success", "msg" : "DB created successfully"}
    except Exception as e:
        ## Handle the exception
        return {"code" : "failure", "msg" : f"DB cannot be created! Actual exception is: {e}"}
    


def create_tables(local_db_name):
    try:
        with sqlite3.connect(local_db_name) as conn:
            c = conn.cursor()

            c.execute('''CREATE TABLE IF NOT EXISTS Users
                        (user_id INTEGER PRIMARY KEY, 
                        username TEXT, 
                        password TEXT)''')

            c.execute('''CREATE TABLE IF NOT EXISTS Searches
                        (search_id INTEGER PRIMARY KEY, 
                        user_id INTEGER, 
                        keyword TEXT,
                        FOREIGN KEY(user_id) REFERENCES Users(user_id))''')

            c.execute('''CREATE TABLE IF NOT EXISTS Results
                        (result_id INTEGER PRIMARY KEY, 
                        search_id INTEGER,
                        pmid TEXT,
                        title TEXT, 
                        abstract TEXT, 
                        body TEXT, 
                        citation TEXT, 
                        processed INTEGER DEFAULT 0,
                        FOREIGN KEY(search_id) REFERENCES Searches(search_id))''')

            conn.commit()
        return {"code" : "success", "msg" : "DB created successfully"}
    except Exception as e:
        ## Handle the exception
        return {"code" : "failure", "msg" : f"DB cannot be created! Actual exception is: {e}"}
 
###############################################################################


def get_unique_pmids(local_db_path, pmids):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT pmid FROM Results")
        pmids_db = [row[0] for row in c.fetchall()]
        unique_pmids = [x for x in pmids if x not in pmids_db]
    return unique_pmids

###############################################################################


def build_search_term(keyword, start_date, end_date, database="pmc"):
    # Format the dates as strings in the correct format
    start_date_str = start_date.strftime("%Y/%m/%d")
    end_date_str = end_date.strftime("%Y/%m/%d")
    if database == "pubmed":
        search_term = f'("{start_date_str}"[Date - Publication] : "{end_date_str}"[Date - Publication]) AND {keyword}[Title/Abstract]'

    elif database == "pmc":
        search_term = ' AND '.join([
            f'("{start_date_str}"[Publication Date] : "{end_date_str}"[Publication Date])',
            f'("{keyword}"[Title] OR "{keyword}"[Abstract])',
            f'(open_access[filter])'])
    return search_term

# def build_search_term_no_review(keywords, start_date, end_date):
#     search_term = ' AND '.join([' NOT '.join([f'("{keywords}"[Title/Abstract])',
#                                 f'(Review[Publication Type])']),
#                                 f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'])
#     return search_term

# def build_search_term_no_review(keywords, start_date, end_date):
#     if "/" in keywords:
#         lst = keywords.split("/")
#         keywords = " OR ".join(f'"{item}"' for item in lst)
#         search_term = ' AND '.join([' NOT '.join([f'("{keywords}"[Title/Abstract])',
#                                 f'(Review[Publication Type])']),
#                                 f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'])
#     else:
#         search_term = ' AND '.join([' NOT '.join([f'("{keywords}"[Title/Abstract])',
#                                 f'(Review[Publication Type])']),
#                                 f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'])
#     return search_term

def build_search_term_no_review(keywords, start_date, end_date):
    if "/" in keywords:
        lst = keywords.split("/")
        keywords = " OR ".join(f'"{item}"' for item in lst)
        search_term = ' AND '.join([' NOT '.join([f'({keywords}[Title/Abstract])',
                                '(Review[Publication Type])']),
                                f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'])
    else:
        search_term = ' AND '.join([' NOT '.join([f'("{keywords}"[Title/Abstract])',
                                '(Review[Publication Type])']),
                                f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'])
    return search_term

###############################################################################


def fetch_with_retry(db, id, retmode, rettype=None, max_tries=5, linear_delay=5):
    Entrez.email = email
    retrying = False
    for i in range(max_tries):
        try:
            handle = Entrez.efetch(
                db=db,
                api_key=os.environ['ENTREZ_API_KEY'],
                id=id,
                retmode=retmode,
                rettype=rettype
            )
            if id == "33451740":
                pdb.set_trace()
            # print(f"pmid = {id}")
            
            # pdb.set_trace()
            print(f"pmid = {id}")
            
            if rettype == "Medline":
                content = Medline.read(handle)
            else:
                content = handle.read()
            handle.close()

            if retrying:
                logger.info(f"After {i+1} attempts SUCCESSFULLY fetched {retmode.upper()} for {db} id: {id}")
            return content
        except (IncompleteRead, HTTPError) as e:
            # print(f"Attempt {i + 1}/{max_tries}: Error fetching {retmode.upper()} for PMC{id}: {e}")
            logger.warning(f"Attempt {i + 1}/{max_tries}: Error fetching {retmode.upper()} for {db} id: {id}: {e}")

            if i + 1 < max_tries:
                time.sleep(linear_delay)
                retrying = True
            else:
                # print(f"Failed to fetch {retmode.upper()} for PMC{id} after {max_tries} attempts")
                logger.warning(f"Failed to fetch {retmode.upper()} for {db} id: {id} after {max_tries} attempts")

                return None
###############################################################################


def get_abstract_from_xml_record(xml_record, id):
    # if id == "32651262":
    #     pdb.set_trace()
    try:
        # parse the XML using beautifulsoup4
        soup = BeautifulSoup(xml_record, features="xml")
        
        abstract = ""
        # abstracts = soup.find_all("abstract")
        # abstracts = soup.find_all("Abstract")
        abstracts = soup.find_all(["Abstract", "abstract"])
        for a in abstracts:
            if not bool(a.attrs) or not a.has_attr("abstract-type"):
                abstract += a.get_text()
                return abstract.lstrip().rstrip()
        return abstract
    except Exception as e:
        print(f"Error extracting abstract content for PM{id}: {e}")
        logger.debug(f"Error extracting abstract content for PM{id}: {e}")
        return ""     
###############################################################################


def get_full_text_from_xml(xml_record, pmc_id):
    try:
        # parse the XML using beautifulsoup4
        soup = BeautifulSoup(xml_record, features="xml")
        
        # get all the paragraphs in the body of the article
        paragraphs = soup.find_all("p")
        if not paragraphs:
            # print(f"Error: no paragraphs found in full-text XML for PMC{pmc_id}")
            logger.debug(f"Error: no paragraphs found in full-text XML for PMC{pmc_id}")
            return None

        # list of strings that indicate sections to ignore
        ignore_sections = ["acknowledgment", "data availability", "conflict of interest",
                           "supplemental", "author contribution", "funding"]

        # extract the text content from the paragraphs, excluding those in the ignored sections
        full_text = ""
        for p in paragraphs:
            parent_section = p.find_parent("sec")
            if parent_section:
                section_title = parent_section.find("title")
                if section_title:
                    title_text = section_title.get_text().lower()
                    # remove number and dot
                    title_text = re.sub(r'^\d+.*?\.\s+', '', title_text)
                    title_text = title_text.strip()  # remove leading/trailing whitespace
                    if any(title_text.startswith(ignore) for ignore in ignore_sections):
                        continue
                # print(title_text)
                # print("#######################################################")
                # print(p.get_text())
                # print("#######################################################")
                full_text += p.get_text() + " "
        return full_text.lower()

    except Exception as e:
        # print(f"Error extracting full-text content for PMC{pmc_id}: {e}")
        logger.debug(f"Error extracting full-text content for PMC{pmc_id}: {e}")
        return None
###############################################################################

def search_keyword(keyword, abstract):
    found = False
    if "/" in keyword:
        keywords = keyword.split("/")
        for keyword in keywords:
            if keyword.lower() in abstract:  
                found = True
                break
    else:
        if keyword.lower() in abstract:
            found = True
    return found

### New v2:
def get_title_abs_from_text(keyword, text_record, xml_record, id):
    try:
        if "AB" in text_record and "TI" in text_record:
            title = text_record["TI"].lower()
            abstract = text_record["AB"].lower()
            if bool(abstract):
                found = search_keyword(keyword, abstract)
                if found:
                    return title, abstract
                else:
                    logger.warning(f"Abstract found in text record for PM{id} but `{keyword}` is not mentioned there, you might wanna take a look when chunks do not look reasonable")
                    return title, abstract
            else:
                logger.info(f"Empty abstract field in text record for PM{id}, trying to get it from xml record")
                abstract = get_abstract_from_xml_record(xml_record, id)
                if bool(abstract):
                    found = search_keyword(keyword, abstract)
                    if found:
                        return title, abstract
                    else:
                        logger.warning(f"Abstract found in xml record for PM{id} but `{keyword}` is not mentioned there, you might wanna take a look when chunks do not look reasonable")
                        return title, abstract
                else:
                    return None, None
        else:
            logger.warning(f"Abstract and/or Title not present for PM{id} in text record")
            return None, None

    except Exception as e:
        # print(f"Error extracting full-text content for PM{id}: {e}")
        logger.error(f"Error extracting full-text content for PM{id}: {e}")
        return None, None
## --------------------------------------------------------------------------##
def get_title_abs_from_text_temp(text_record, xml_record, id):
    try:
        if "AB" in text_record and "TI" in text_record:
            title = text_record["TI"].lower()
            abstract = text_record["AB"].lower()
            if bool(abstract):
                return title, abstract
            else:
                logger.info(f"Empty abstract field in text record for PM{id}, trying to get it from xml record")
                abstract = get_abstract_from_xml_record(xml_record, id)
                if bool(abstract):
                    return title, abstract
                else:
                    return None, None
        else:
            logger.warning(f"Abstract and/or Title not present for PM{id} in text record")
            return None, None

    except Exception as e:
        logger.error(f"Error extracting full-text content for PM{id}: {e}")
        return None, None    
###############################################################################


def get_citation_str(text_record, id):
    citation_str = ""

    authors = text_record.get('AU', [])
    citation_str = ", ".join(authors) + ". "

    title = text_record.get('TI', '')
    if title:
        citation_str += title
        if not title.endswith('.'):
            citation_str += ". "
        else:
            citation_str += " "

    journal = text_record.get('JT', '')
    if journal:
        citation_str += journal + ". "

    pub_date = text_record.get('DP', '')
    if pub_date:
        # Extract the year from the 'pub_date' field
        year = pub_date.split(' ')[0]
        if year:
            citation_str += year + ";"
            citation_str += " "

    volume = text_record.get('VI', '')
    if volume:
        citation_str += volume
        # citation_str += " "

    issue = text_record.get('IP', '')
    if issue:
        citation_str += f"({issue}). "
        # citation_str += " "

    pages = text_record.get('PG', '')
    if pages:
        citation_str += f":{pages}. "
        # citation_str += " "

    doi = text_record.get('AID', '')
    if doi:
        doi_elements = [elem for elem in doi if '[doi]' in elem]
        if doi_elements:
            citation_str += f"{doi_elements[0]}."
        else:
            citation_str += f" {doi}."

    if bool(citation_str):
        return citation_str
    else:
        citation_str += f"citation info missing for PM{id}."
        return citation_str
###############################################################################


def insert_users(local_db_path, users):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.executemany('''INSERT INTO Users(username, password) VALUES(?, ?)''', users)
        conn.commit()

# # use it like this
# users = [('testuser1', 'testpassword1'), ('testuser2', 'testpassword2')]
# insert_users('local_db_name.db', users)

def insert_search(local_db_path, user_id, keyword):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO Searches(user_id, keyword) VALUES(?, ?)''', (user_id, keyword))
        search_id = c.lastrowid
        conn.commit()
        return search_id  # return the id of the search


def insert_results(local_db_path, search_id, results_df):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        # Convert the DataFrame to a list of tuples
        results = [tuple(x) for x in results_df.values]
        # add the search_id to each result
        results_with_search_id = [(search_id, *result) for result in results]
        c.executemany('''INSERT INTO Results(search_id, pmid, title, abstract, body, citation, processed) 
                         VALUES(?, ?, ?, ?, ?, ?, ?)''', results_with_search_id)
        conn.commit()

# # use it like this
# # assuming keyword is the search keyword and results_df is a pandas DataFrame
# search_id = insert_search('local_db_name.db', 1, keyword)  # hardcoded user_id 1
# insert_results('local_db_name.db', search_id, results_df)
###############################################################################


def pmid2pmcid(email, pmids, max_tries=5, linear_delay=5):
    Entrez.email = email
    pmid2pmcid = {}
    for pmid in pmids:        
        for i in range(max_tries):
            try:
                handle = Entrez.elink(dbfrom="pubmed", db="pmc", linkname="pubmed_pmc", id=pmid, retmode="text")
                handle_read = handle.read()
                handle.close()
                break
            except (IncompleteRead, HTTPError) as e:
                # print(f"Attempt {i + 1}/{max_tries}: Error retrieving IDs for PM{pmid}: {e}")
                logger.warning(f"Attempt {i + 1}/{max_tries}: Error retrieving IDs for PM{pmid}: {e}")

                if i + 1 < max_tries:
                    time.sleep(linear_delay)
                else:
                    # print(f"Failed to retrieve IDs for PM{pmid} after {max_tries} attempts")
                    logger.warning(f"Failed to retrieve IDs for PM{pmid} after {max_tries} attempts")
                    # logger.warning(f"Failed to fetch {retmode.upper()} for PMC{pmcid} after {max_tries} attempts")
        root = ET.fromstring(handle_read)
        pmcid = ""
        for link in root.iter('Link'):
            for id in link.iter('Id'):
                pmcid = id.text
        if not pmcid == "":
            pmid2pmcid[pmid] = pmcid
        else:
            pmid2pmcid[pmid] = "-"
    return pmid2pmcid 
###############################################################################


def generate_visitor_body(footer_cut, header_cut, parts):
    def visitor_body(text, cm, tm, fontDict, fontSize):
        y = tm[5]
        if y > footer_cut and y < header_cut:
            parts.append(text)
    return visitor_body
###############################################################################

def remove_unwanted_spaces(text):
    # The regex to match
    pattern = re.compile(r"(?<=\s|\)|\])(\b[\w-]+\s+et al\.\s\(\d{4}\)|\(\s*[^\d\(]*\d{4}(?:;[^\d\(]*\d{4})*\s*\)|\((?:\d+(?:-\d+)?)(?:,\s*(?:\d+(?:-\d+)?))*\)|\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\])")

    # Replace the citations
    text_without_citations = re.sub(pattern, '', text)

    # Post-process text to remove multiple spaces
    text_without_citations = re.sub(' +', ' ', text_without_citations)

    # Trim spaces around punctuation
    text_without_citations = re.sub(r'\s+([,.])', r'\1', text_without_citations)
    
    return text_without_citations.strip()

def iter_paragraphs(paragraph):
    """Iterates over a paragraph and its children to extract all text."""
    if paragraph.text:
        yield paragraph.text
    for child in paragraph:
        yield from iter_paragraphs(child)
        if child.tail:
            yield child.tail

###############################################################################


# ## v3
def extract_text_form_pdf(xml_dir, local_db_path):  
    # pdb.set_trace()
    ## Extract the data from local DB in a df 
    db_table = "Results"
    df = fetch_all_data_as_df(local_db_path, db_table)
        
    ## Define the namespace
    namespaces = {'ns': 'http://www.tei-c.org/ns/1.0'}
        
    ## Get a list of xml files and extract pmids from file names
    xml_fles = glob.glob(xml_dir + "/*.tei.xml")
    pmids = [os.path.basename(file).rstrip(".tei.xml") for file in xml_fles]
    
    ## Create a list of headers upon hitting which code must stop extracting
    stop_headers = ['acknowledgement', 'references', 'funding', 'availability']
    
    for pmid in pmids: 
        tree = ET.parse(xml_dir + '/' + pmid + '.tei.xml')
        
        ## Build a dictionary to map children to their parent
        parent_map = {c: p for p in tree.iter() for c in p}
        
        ## Find all paragraph elements
        paragraphs = tree.findall('.//ns:p', namespaces)
        
        ## Extract and print the text from each paragraph that's not inside an 'acknowledgement' div
        all_text = ""
        stop_flag = False
        for p in paragraphs:
            parent = parent_map.get(p)
            skip = False
            while parent is not None:
                if parent.tag == '{http://www.tei-c.org/ns/1.0}profileDesc':
                    for abstract in parent.findall('.//ns:abstract', namespaces):
                        if abstract is not None:
                            skip = True  # Skip the abstract section
                            break  # Stop going up the tree
                elif parent.tag == '{http://www.tei-c.org/ns/1.0}div' and parent.attrib.get('type') in stop_headers:
                    stop_flag = True
                    break
                parent = parent_map.get(parent)
    
            if stop_flag:
                break
            elif skip:
                continue
            else:
                text = ''.join(iter_paragraphs(p))
                if len(text.split()) < 10:
                    continue
                all_text += text + "\n"
        
        all_text = all_text.replace("\n", " ").lower()
        all_text = remove_unwanted_spaces(all_text)
        # pdb.set_trace()

        if all_text:
            df.loc[df.pmid == pmid, ['body']] = all_text
            df.loc[df.pmid == pmid, ['processed']] = 0
    return df
###############################################################################


def extract_title(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Define the namespace
    namespaces = {'ns': 'http://www.tei-c.org/ns/1.0'}

    # Find the titleStmt tag and then find the title tag within it
    titleStmt = root.find('.//ns:titleStmt', namespaces)
    if titleStmt is not None:
        title = titleStmt.find('ns:title', namespaces)
        if title is not None and 'type' in title.attrib and title.attrib['type'] == 'main':
            return title.text

    return "Title not found"

###############################################################################


def pmc_text_downloader(keyword, start_date, end_date, max_papers, local_db_path, user_id):

    start = time.time()
        
    ## Save the date strings to use later in the log
    start_date_str = start_date
    end_date_str = end_date

    ## Convert the date strings into strptime objects
    start_date = datetime.strptime(start_date, "%Y/%m/%d")
    end_date = datetime.strptime(end_date, "%Y/%m/%d")

    ## set your email address for Entrez API requests (required)
    Entrez.email = email

    ## specify the author's name and search terms
    # keyword = 'dosage compensation'

    ## database selection
    database = "pubmed"
    # database = "pmc"
    
    # pdb.set_trace()
        
    search_term = build_search_term_no_review (keyword, start_date_str, end_date_str)    
    # search_term = "((dosage compensation[Title/Abstract]) AND (Drosophila[Title/Abstract])) NOT (Review[Publication Type])"
    
    ## search the Entrez database and retrieve the results
    handle = Entrez.esearch(db=database,
                            term=search_term,
                            retmax=int(max_papers),
                            sort="pub date",
                            api_key=os.environ['ENTREZ_API_KEY'])

    record = Entrez.read(handle)

    ## get the PMIDs for each result and covert them to pmcids
    pmids = record["IdList"]
    handle.close()
    
    ## Temporary pmids


    pmid_pmcid_dict = pmid2pmcid(email, pmids)
    # ## dump dict to disk
    # with open('pmid_pmcid_dict.pkl', 'wb') as f:
    #     pickle.dump(pmid_pmcid_dict, f)        
    ## read dict from disk
    # with open('pmid_pmcid_dict.pkl', 'rb') as f:
    #     pmid_pmcid_dict = pickle.load(f)
            
    
    # Print the number of matching articles
    if len(pmid_pmcid_dict) != 0:
        message = f"Found {len(pmid_pmcid_dict)} articles published between {start_date_str} and {end_date_str} in {database}"
        logger.info(message)
        logger.info("Checking before appending to db if the pmids/pmcids are unique")
    else:
        message = f"There are 0 articles published between {start_date_str} and {end_date_str} in {database}"
        logger.info(message)
        logger.info("Therefore nothing to append in {local_db_path} database")
        return {"code" : "exit", "msg" : message}
    
    ## first check if the pmids have duplicates, only process unique ones
    ## here the keys in pmid_pmcid_dict are pmids
    uniqe_pmids = get_unique_pmids(local_db_path, list(pmid_pmcid_dict.keys()))
    
    ## Assuming keywords is the search keywords get the search_id for user with user_id
    search_id = insert_search(local_db_path, user_id, keyword)
    if not bool(uniqe_pmids):
        # print("No new pmcids are found, therefore nothing will be appended in local db, pmc downloader will exit now")

        logger.critical("No new pmids are found, therefore nothing will be appended in local db, article downloader will exit now")
        return {"code" : "exit", "msg" : "No new pmids are found, check log for more information"}
    else:
        pmids = uniqe_pmids
        logger.info(f"{len(pmids)} new pmcids are found, now attempting to append in local db")
    
    # pmids = ["34694912", "34618146", "32762846", "32432549"]
    
    for pmid in pmids:       
        ## check if the pmid has a pmcid associated with it, if yes then 
        ## fetch the full text otherwise fetch the abstract and title
        pmcid = pmid_pmcid_dict[pmid]
        full_text = "" 
        
        ## Retrieve the article's xml records and Medline text using PMID
        text_record_pmid = fetch_with_retry(db="pubmed", 
                                       id=pmid,
                                       retmode="text", 
                                       rettype="Medline")
        xml_record_pmid = fetch_with_retry(db="pubmed", 
                                        id=pmid, 
                                        retmode="xml")
                
        if pmcid != "-":
            ## Retrieve the article's xml records and Medline text using PMCID
            text_record = fetch_with_retry(db="pmc", 
                                            id=pmcid, 
                                            retmode="text", 
                                            rettype="Medline")
            xml_record = fetch_with_retry(db="pmc", 
                                          id=pmcid, 
                                          retmode="xml")
            if xml_record and text_record:
                full_text = get_full_text_from_xml(xml_record, pmcid)
                    
        ## If both xml and text records are NOT empty, then prepare to insert the record in local DB                 
        if xml_record_pmid and text_record_pmid:
            article_title, article_abstract = get_title_abs_from_text(keyword,
                                                                      text_record_pmid, 
                                                                      xml_record_pmid, 
                                                                      pmid)
            citation_str = get_citation_str(text_record_pmid, pmid)
            
            if article_title and article_abstract and citation_str:
                if full_text:
                    message = f"Found full-text content for PM{pmid} and will be added to DB"
                else:
                    message = f"Could not find full-text content for PM{pmid} hence only Abstract will be added to DB"
                logger.info(message)
                try:                        
                    ## Insert the result into the Results table  
                    results_df = pd.DataFrame({'pmid' : [pmid], 
                                                'title' : [article_title], 
                                                'abstract' : [remove_newline_multiple_spaces(article_abstract)], 
                                                'body' : [full_text] if full_text == "" else [remove_newline_multiple_spaces((remove_unwanted_spaces(full_text)))],
                                                'citation' : [citation_str],
                                                'processed': [0]})  
                    insert_results(local_db_path, search_id, results_df)  
                    logger.info(f"Sucessfully inserted data into DB for PM{pmid}")
                except:
                    message = f"Could not append result/abstract/title/citation for PM{pmid}"
                    # print(message)
                    logger.info(message)
                    continue
            else:
                if not article_title:
                    message = f"Title NOT found for PM{pmid}, hence will be ignored"
                elif not article_abstract:
                    message = f"Abstract NOT found for PM{pmid}, hence will be ignored"
                elif not citation_str:
                    message = f"Citation NOT found for PM{pmid}, hence will be ignored"                    
                logger.warning(message)
        else:
            if not xml_record:
                # print(f"Empty XML record for PM{pmid}")
                logger.info(f"Empty XML record for PM{pmid}")
            elif not text_record:
                # print(f"Empty TEXT record for PM{pmid}")
                logger.info(f"Empty TEXT record for PM{pmid}")
            elif not xml_record and not text_record:
                # print(f"Empty XML and TEXT records for PM{pmid}")
                logger.info(f"Empty XML and TEXT records for PM{pmid}")

    end = time.time()
    # print("\n\n")
    # print(f"Time elapsed during article download = {end - start}")
    msg = "Time elapsed during article download"
    log_elapsed_time(start, end, msg)
    # pdb.set_trace()
    return {"code" : "success", "msg" : "Articles fetched successfully, check log for more information"}

###############################################################################

## Updated for pinecone-client version ≥ 3.0.0
def get_pinecone_index():
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])  
    index = pc.Index('namespaces-in-paper')
    return index
###############################################################################

## Compatible with pinecone-client version ≥ 3.0.0
def delete_vectors_in_namespace(namespace):
    index = get_pinecone_index()
    index.delete(deleteAll=True, namespace=namespace)
###############################################################################

def create_tmp_dir(directory):
    ## Create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory) 


def create_files_from_excel_columns(directory, df):
    ## Save abstracts and body texts in separate text files
    idx_ignored_record = []
    idx_ignored_abs = []
    idx_ignored_body = []
    count_body = 0
    count_abs = 0
    for i, (pmid, abstract, body, citation) in enumerate(zip(df['pmid'], df['abstract'], df['body'], df['citation'])):
        ignore_record = True
        citation_flag = False
        len_abs = len(abstract.split())
        len_res = len(body.split())
        keep_abs = False
        keep_res = False
        
        if len_abs <= 80:
            if len_res > 300:
                keep_abs = True
                keep_res = True 
                logger.info(f"For PubMed id = {pmid}, though abstract text has {len_abs} words(<80) but because body has {len_res} words (>300), the abstract text is also kept")
        else:
            keep_abs = True
            if len_res > 300:
                keep_res = True
                
        if keep_abs:
            count_abs += 1
            abs_file = f"{directory}/abstract_{i+1}.txt"
            with open(abs_file, "w") as f:
                f.write(abstract)
            citation_flag = True
        else:
            logger.info(f"For PubMed id = {pmid}, abstract text has {len_abs} words, which is less than 80 hence abstract ignored")
            idx_ignored_abs.append(i)
            
        if keep_res:
            count_body += 1
            res_file = f"{directory}/body_{i+1}.txt"
            with open(res_file, "w") as f:
                f.write(body)
            citation_flag = True
        else:
            logger.info(f"For PubMed id = {pmid}, body text has {len_res} words, which is less than 300 hence body is ignored")
            idx_ignored_body.append(i)                         
        
        if citation_flag:
            citation_file = f"{directory}/citation_{i+1}.txt"
            with open(citation_file, "w") as f:
                f.write(citation)
            ignore_record = False
            
        if ignore_record:
            logger.info(f"For PubMed id = {pmid}, abstract text has {len_abs} words (< 80) and body text has {len_res} words (< 300) hence entire record is ignored")
            idx_ignored_record.append(i)
               
    logger.info("START Summary create_files_from_excel_columns()>>")
    logger.info(f"Total abstact texts considered = {count_abs}")
    logger.info(f"Total body texts considered = {count_body}")
    logger.info("END Summary create_files_from_excel_columns()>>")
###############################################################################


def sort_list(document_list):
    sorted_documents = sorted(
        document_list, key=lambda doc: doc.metadata['source'])
    # for doc in sorted_documents:
    #     print(doc)
    #     print("################################################")
    return sorted_documents
###############################################################################

def get_matching_meta(record_text, citations, number_txt_list):
    # pdb.set_trace()
    sub_str = record_text.metadata["source"].split("_")[-1]
    try:
        idx = number_txt_list.index(sub_str)
    except ValueError:
        print(f"'{sub_str}' not found in the list")
    return {'matching_citation' : citations[idx].page_content, 'paper_id' : int(sub_str.split(".")[0])}
###############################################################################

def update_all_rows(local_db_path, table_name, column_name, new_value):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.execute(f"UPDATE {table_name} SET {column_name} = ?", (new_value,))
        conn.commit()
###############################################################################

## Updated for pinecone-client version ≥ 3.0.0
def get_num_vectors_in_namespace(namespace):
   
    # Connect to an existing index
    index = get_pinecone_index()
    
    # Get number of vectors in the namespace
    index_stats = index.describe_index_stats()
    num_vectors = index_stats.get('namespaces')[namespace].vector_count
    
    # logger.info(f"Number of vectors in namespace {namespace}: {num_vectors}")   
    return num_vectors
###############################################################################

def remove_newline_multiple_spaces(string):
    string = string.replace("\n", " ")
    string = re.sub(' +', ' ', string).strip()
    return string
###############################################################################      
       
## For WeiseEule paper revision
def return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len):
    embeds = []
    for text in lines_batch:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        # pdb.set_trace()
        outputs = outputs.last_hidden_state.mean(dim=1).numpy()            
        outputs_padded = np.pad(np.squeeze(outputs), (0, openai_vec_len - outputs.shape[1]), 'constant', constant_values=0)
        embeds.append(outputs_padded.tolist())
    return embeds 

    
def get_text_embeddings(text_type, lines_batch, embedd_model, openai_vec_len):
    if embedd_model == "openai":
        # res = openai.Embedding.create(
        #     input=lines_batch, engine=EMBEDDING_MODEL)
        embeds = []
        for text in lines_batch:
            res = syncClient.embeddings.create(input=text, model=EMBEDDING_MODEL)
            embeds.append(res.data[0].embedding)

    elif embedd_model == "biobert":
        biobert_model = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = BertTokenizer.from_pretrained(biobert_model)
        model = BertModel.from_pretrained(biobert_model)
        max_length = 512
        
        embeds = return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len)

    elif embedd_model == "MedCPT":
        if text_type == 'chunk':
            max_length = 512
            logger.info(f"Inside get_text_embeddings(), text_type = {text_type} and max_length = {max_length}")
            model_name = "ncbi/MedCPT-Article-Encoder"
        else:
            max_length = 64
            logger.info(f"Inside get_text_embeddings(), text_type = {text_type} and max_length = {max_length}")
            model_name = "ncbi/MedCPT-Query-Encoder"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embeds = return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len)
      
    return embeds
###############################################################################      


def preprocess_data_qa(local_db_path, 
                       table_name, 
                       namespace, 
                       embedd_model, 
                       chunk_size, 
                       data_dir="tmp"):
    logger.info("Pre-processing and inserting vectors into pinecone db")
    # print("Pre-processing and inserting vectors into pinecone db")

    # Read the data from local db to a df
    df = fetch_all_data_as_df(local_db_path, table_name)
    
    # Only extract those rows from df that are not yet processed
    df = df[df.processed == 0]
    
    ## create a small df with 10 paper records temporarily
    # df = df[:10]

    # Now create files from the columns of this dfcd 
    if df.empty:
        msg = "There is no new record to process, execution is terminated"
        logger.critical(msg)
        return {"code" : "failure", "msg" : msg} 
    else:
        try:
            create_tmp_dir(data_dir)
        except:
            msg = f"{os.path.abspath(data_dir)} cannot be created!"
            logger.critical(msg)
            return {"code" : "failure", "msg" : msg}        
        try:
            create_files_from_excel_columns(data_dir, df)
        except:
            msg = f"Could not create files in {os.path.abspath(data_dir)}"
            logger.critical(msg)
            return {"code" : "failure", "msg" : msg}
    
    # load results data
    # loader = DirectoryLoader('./abs_res/', glob='**/result*.txt')
    loader = DirectoryLoader('./' + data_dir + '/', glob='**/body*.txt')
    results = loader.load()
    logger.info(f'You have {len(results)} body texts')

    logger.info(f'Spliliting long body texts into smaller chunks of size = {chunk_size}')

    ## Pre-process the body text of each record
    res_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0)
    
    result_chunks_indexed = []
    # split the text in result files into managable size chunks for openai api calls
    for result in results:
        result.page_content = remove_newline_multiple_spaces(result.page_content)
        results_splitted = res_splitter.split_documents([result])

        for chunk_idx, chunk in enumerate(results_splitted):
            chunk.metadata['chunk_id'] = chunk_idx
        result_chunks_indexed.append(results_splitted)
    
    ## From list of lists to list
    result_chunks_indexed = [item for sublist in result_chunks_indexed for item in sublist]
        
    logger.info(f'After spliliting body into smaller texts you have {len(result_chunks_indexed)} texts')


    # load abstract data
    loader = DirectoryLoader('./' + data_dir + '/', glob='**/abstract*.txt')
    abstracts = loader.load()
    
    ## Index all the abstract text with -1, since they are not splitted
    # for a paper first chunk is its abstract with chunk id = -1
    for abstract in abstracts:
        abstract.page_content = remove_newline_multiple_spaces(abstract.page_content)
        abstract.metadata['chunk_id'] = -1
        

    # load citation data
    loader = DirectoryLoader('./' + data_dir + '/', glob='**/citation*.txt')
    citations = loader.load()

    # sort the results, abstracts and citations to achive one-to-one correspondence
    result_chunks_indexed = sort_list(result_chunks_indexed)
    abstracts = sort_list(abstracts)
    citations = sort_list(citations)

    # Now merge results and abstracts
    all_texts = abstracts + result_chunks_indexed
    logger.info(f'After merging abstracts with body texts you have {len(all_texts)} texts')

    # extract only part of filename from citation metadata
    pattern = r"\d+\.txt"
    citation_list = [record.metadata for record in citations]
    number_txt_list = [re.search(pattern, item['source']).group(0)
                        for item in citation_list]

    # all_texts_citated = []
    # for record in all_texts:
    #     matching_citation = get_matching_meta(
    #         record, citations, number_txt_list)
    #     record.metadata["citation"] = matching_citation
    #     all_texts_citated.append(record)
        
    all_texts_citated = []
    for record in all_texts:
        # pdb.set_trace()
        matching_citation = get_matching_meta(record, citations, number_txt_list)
        record.metadata["citation"] = matching_citation['matching_citation']
        record.metadata["paper_id"] = matching_citation['paper_id']
        all_texts_citated.append(record)

    # # only_text = [record.page_content for record in all_texts_citated]
    # # only_meta = [record.metadata["citation"] for record in all_texts_citated]
    
    # # pdb.set_trace()

    # # # initialize pinecone and connect to an index
    # # index = get_pinecone_index()

    # # push vectors in pinecone DB
    # # push_vectors_into_pinecone(index, only_text, only_meta, EMBEDDING_MODEL)
    
    # with open('all_texts_citated.pkl', 'rb') as f:
    #     all_texts_citated = pickle.load(f)
    # pdb.set_trace()  
    try:
        # push_vectors_into_pinecone(index,
        #                            only_text,
        #                            only_meta,
        #                            namespace,
        #                            embedd_model)
        push_vectors_into_pinecone(all_texts_citated,
                                   namespace,
                                   embedd_model)
                
        update_all_rows(local_db_path,
                        table_name,
                        column_name="processed",
                        new_value=1)
        
        msg = f"Vectors pushed successfully into pinecone DB"
        logger.info(msg)
        return {"code" : "success", "msg" : msg} 
    except Exception as e:
        msg = f"Error pushing the vectors in to pinecone, actual exception is: {e}"
        logger.critical(msg)
        pdb.set_trace()
        return {"code" : "failure", "msg" : msg}  
###############################################################################

## Compatible with pinecone-client version ≥ 3.0.0
def push_vectors_into_pinecone(all_texts_citated, namespace, embedd_model):
    # namespace += '_' + embedd_model
    # pdb.set_trace()
    logger.info("Inside push_vectors_into_pinecone()")
    # print("Inside push_vectors_into_pinecone()")
    
    # initialize pinecone and connect to an index
    index = get_pinecone_index()
    
    ## Get the existing number of vectors in a namespace
    index_stats = index.describe_index_stats()
    if namespace in index_stats["namespaces"]:
        offset = get_num_vectors_in_namespace(namespace)
        logger.info(f"Number of vectors in namespace {namespace}: {offset}")
        # print(f"Number of vectors in namespace {namespace}: {offset}")
    else:
        offset = 0
        logger.info(f"The namespace {namespace} does not yet exist, will be created, hence in the beginning number of vectors in namespace {namespace} is: {offset}")
        # print(f"The namespace {namespace} does not yet exist, will be created, hence number of vectors in namespace {namespace} will be: {offset}")
                          
    batch_size = 32  # process everything in batches of 32
    
    for i in tqdm(range(0, len(all_texts_citated), batch_size)):   
        # pdb.set_trace()                   
        # print(f"\ni = {i}\n")
        
        # set end position of batch
        i_end = min(i+batch_size, len(all_texts_citated))

        # get batch of lines and IDs
        records_batch = all_texts_citated[i: i+batch_size]

        # ids_batch = [str(n) for n in range(i, i_end)]
        ids_batch = [str(n + offset + 1) for n in range(i, i_end)]        

        texts_batch = [record.page_content  for record in records_batch]
        # citations_batch = [record.metadata['citation']  for record in records_batch]
        
        if bool(texts_batch):
            ## Create embeddings 
            # embeds = get_text_embeddings(texts_batch, embedd_model, openai_vec_len=1536)
            embeds = get_text_embeddings('chunk', texts_batch, embedd_model, openai_vec_len=1536)
            
            # prep metadata and upsert batch
            meta = [{'text': text} for text in texts_batch]
            cite = [{'citation': record.metadata['citation']} for record in records_batch]
            paper_id = [{'paper_id': record.metadata['paper_id']} for record in records_batch]
            chunk_id = [{'chunk_id': record.metadata['chunk_id']} for record in records_batch]
            
            
            
            if len(meta) == len(cite):
                for i, dict_item in enumerate(meta):
                    dict_item['citation'] = cite[i]['citation']
                    dict_item['paper_id'] = paper_id[i]['paper_id']
                    dict_item['chunk_id'] = chunk_id[i]['chunk_id']
            else:
                msg = "Lists have different lengths, cannot add citations to pinecone meta. Run terminated."
                logger.info(msg)
                return {"code" : "failure", "msg" : msg}
            
            to_upsert = zip(ids_batch, embeds, meta)
            # pdb.set_trace()
        else:
            ## Empty text in records_batch
            # import sys
            # sys.exit('Empty text in records_batch')
            msg = "Empty text in records_batch"
            return {"code" : "failure", "msg" : msg}

        ## upsert to Pinecone            
        # logger.info(f"Pushing {len(ids_batch)} vectors into Pinecone")     
        logger.info(f"Pushing {len(ids_batch)} vectors into Pinecone")
        try:
            index.upsert(vectors=list(to_upsert), namespace=namespace)
        except Exception as e:
            msg = f"Error pushing the vectors in to pinecone, actual exception is: {e}"
            
            
    # pdb.set_trace()
            
    # Close the index and return (NOT REQUIRED in pinecone-client version ≥ 3.0.0)
    # index.close()
###############################################################################


def fetch_and_display_all_data(local_db_name, db_table):
    conn = sqlite3.connect(local_db_name)
    c = conn.cursor()
    c.execute(f"SELECT * FROM {db_table}")
    all_data = c.fetchall()
    conn.close()

    print(f"All data in the {db_table} table:")
    for row in all_data:
        print(row)
        # pdb.set_trace()
############################################################################### 


def fetch_all_data_as_df(local_db_name, db_table):
    conn = sqlite3.connect(local_db_name)
    df = pd.read_sql_query(f"SELECT * FROM {db_table}", conn)
    conn.close()    
    return df        
###############################################################################


def check_table(local_db_name, db_table):
    with sqlite3.connect(local_db_name) as conn:
        c = conn.cursor()

        # Get the table information
        c.execute(f'PRAGMA table_info({db_table});')
        info = c.fetchall()

        if not info:
            print(f"No table named {db_table} exists in the database.")
        else:
            print(f"Table {db_table} exists with the following structure:")
            for column in info:
                print(column)
###############################################################################
def find(lst, key):
     for i, x in enumerate(lst):
         if x == key:
             break
     return i


def update_processed(local_db_path, value):
    try:
        with sqlite3.connect(local_db_path) as conn:
            c = conn.cursor()

            # c.execute('''UPDATE Results SET processed = 1''')
            c.execute('''UPDATE Results SET processed = ?''', (value,))
            conn.commit()
        return {"code" : "success", "msg" : "Column 'processed' updated successfully"}
    except Exception as e:
        ## Handle the exception
        return {"code" : "failure", "msg" : f"Column cannot be updated! Actual exception is: {e}"}
    
    
def update_processed_list(local_db_path, df):
    ids = (df.index + 1).tolist()
    values = df['processed'].tolist()
    try:
        if len(ids) != len(values):
            raise ValueError("ids and values must have the same length")

        with sqlite3.connect(local_db_path) as conn:
            c = conn.cursor()

            for id, value in zip(ids, values):
                # print(f"(id, value) = {(id, value)}")
                c.execute('''UPDATE Results SET processed = ? WHERE result_id = ?''', (value, id))
            conn.commit()
        return {"code" : "success", "msg" : "Column 'processed' updated successfully"}
    except Exception as e:
        ## Handle the exception
        return {"code" : "failure", "msg" : f"Column cannot be updated! Actual exception is: {e}"}
    

def replace_table_contents(local_db_path, table_name, df):
    try:
        with sqlite3.connect(local_db_path) as conn:
            c = conn.cursor()
            
            # Step 1: Delete all rows from the table
            c.execute(f'DELETE FROM {table_name}')
            
            # Step 2: Insert new rows into the table
            # Generate the placeholders for the SQL query based on the number of columns
            placeholders = ','.join(['?'] * len(df.columns))
            
            # Convert DataFrame to list of tuples
            data = [tuple(row) for row in df.itertuples(index=False)]
            
            # Insert the data
            c.executemany(f'INSERT INTO {table_name} VALUES ({placeholders})', data)
            
            conn.commit()
            
        return {"code" : "success", "msg" : "Table contents replaced successfully"}
    except Exception as e:
        # Handle the exception
        return {"code" : "failure", "msg" : f"Table contents cannot be replaced! Actual error: {e}"}   
###############################################################################    

    
# async def check_article_contents(namespace, pmid):
#     """
#     Checks whether an abstract and/or full-body text is available for a given PubMed ID in the Results table of a SQLite database.

#     Parameters:
#     - namespace (str): The selected namespace.
#     - pmid (str): The PubMed ID to search for.

#     Returns:
#     - str: A message indicating the availability of the text.
#     """
#     # From namespace get matching sql DB and convert into a dict
#     map_table = pd.read_excel("Local_DB/MAP_namespace_sql_DB.xlsx")
#     result_dict = pd.Series(map_table.sql_DB.values, index=map_table.namespace).to_dict()
#     if not namespace in result_dict:
#         error_message = f"DB corresponding to the namespace `{namespace}` is missing from `Local_DB` folder."
#         logger.info("Inside check_article_contents()")
#         logger.info("error_message = {error_message}")
#         yield {"code": "failure", "msg": error_message}
#         # return
    
#     # Connect to the SQLite database
#     db_path = "Local_DB/" +  result_dict[namespace]
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # SQL query to select abstract and body for the given pmid
#     query = "SELECT abstract, body FROM Results WHERE pmid = ?"
    
#     try:
#         # Execute the query
#         cursor.execute(query, (pmid,))
#         result = cursor.fetchone()

#         # Check if the pmid is found in the database
#         if result:
#             abstract, body = result
#             if abstract and body:
#                 msg = "Both abstract and body text are available"
#             elif abstract:
#                 msg = "Only abstract text is available"
#             elif body:
#                 msg = "Only body text is available"
#             else:
#                 msg = "No abstract or body text is available"
#             yield {"code": "success", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
#             # return

#         else:
#             msg = "No record found for the provided PMID"
#             yield {"code": "not found", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
#             # return
        

#     except sqlite3.Error as e:
#         msg =  f"An error occurred: {e}"
#         yield {"code": "failure", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
#         # return
#     finally:
#         # Close the database connection
#         conn.close()

## V2: no namespace:local db mapping required
async def check_article_contents(namespace, pmid):
    """
    Checks whether an abstract and/or full-body text is available for a given PubMed ID in the Results table of a SQLite database.

    Parameters:
    - namespace (str): The selected namespace.
    - pmid (str): The PubMed ID to search for.

    Returns:
    - str: A message indicating the availability of the text.
    """
    ## Find the correct local DB corresponding to the namespace
    local_DB_dir = "Local_DB"
    local_db_path = './' + local_DB_dir + '/' + namespace + '.db'
    if not os.path.isfile(local_db_path):
        error_message = f"DB corresponding to the namespace `{namespace}` is missing from `Local_DB` folder."
        yield {"code": "failure", "msg": error_message}

    # Connect to the SQLite database
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()

    # SQL query to select abstract and body for the given pmid
    query = "SELECT abstract, body FROM Results WHERE pmid = ?"

    try:
        # Execute the query
        cursor.execute(query, (pmid,))
        result = cursor.fetchone()

        # Check if the pmid is found in the database
        if result:
            abstract, body = result
            if abstract and body:
                msg = "Both abstract and body text are available"
            elif abstract:
                msg = "Only abstract text is available"
            elif body:
                msg = "Only body text is available"
            else:
                msg = "No abstract or body text is available"
            yield {"code": "success", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
            # return

        else:
            msg = "No record found for the provided PMID"
            yield {"code": "not found", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
            # return
        

    except sqlite3.Error as e:
        msg =  f"An error occurred: {e}"
        yield {"code": "failure", "msg": f"<b>namespace:</b> {namespace}<p></p><b>pmid:</b> {pmid}<p></p><b>result:</b> {msg}."}
        # return
    finally:
        # Close the database connection
        conn.close()

###############################################################################
###############################################################################
