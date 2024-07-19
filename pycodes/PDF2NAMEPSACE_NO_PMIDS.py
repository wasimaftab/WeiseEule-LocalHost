#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:31:25 2024

@author: wasim
"""
#!/usr/bin/bash
# make sure to activate proper conda environment by running `mamba activate qa`

##------------------------------ PDF 2 xml ----------------------------------##
import time
import subprocess
import pdb
from tkinter import filedialog as fd
import sys
## Select the folder with your pdf files
pdf_dir = fd.askdirectory(title = "Select the folder with your pdf files")
if not bool(pdf_dir):
    sys.exit("No folder selected")

# Start GROBID server as a background process
subprocess.Popen('bash -i -c grobid-run', shell=True)

# Wait for a few seconds to ensure the server is up and running
time.sleep(10)  # Adjust this based on how long your server typically takes to start

# Now continue with the client
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="../grobid_client_python-master/config.json")

client.process("processFulltextDocument", 
                input_path=pdf_dir, 
                output=pdf_dir + "_xml",
                n=10,  # may want to adjust this based on the cores available
                force=True)

##-------------------- Extract text from xml into DB ------------------------##

import glob
import os 
import re
import xml.etree.ElementTree as ET
import PMC_downloader_Utils as pmcd
import pandas as pd
import sqlite3
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# cwd = os.path.dirname(os.path.realpath(__file__))

def delete_table_contnts(local_db_path, table_name):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.execute(f'DELETE FROM {table_name}')
        conn.commit()
###############################################################################

def create_files_from_excel_columns(directory, df):
    ## Save abstracts and body texts in separate text files
    count_body = 0
    for i, (citation, body) in enumerate(zip(df['citation'], df['body'])):
        ignore_record = True
        citation_flag = False
        len_res = len(body.split())
        keep_res = False
        
        if len_res > 300:
            count_body += 1
            res_file = f"{directory}/body_{i+1}.txt"
            with open(res_file, "w") as f:
                f.write(body)
            
            citation_file = f"{directory}/citation_{i+1}.txt"
            with open(citation_file, "w") as f:
                f.write(citation)
        else:
            print(f"For citation = {citation}, body text has {len_res} words, which is less than 300 hence body is ignored")
                    
    print("START Summary create_files_from_excel_columns()>>")
    print(f"Total body texts considered = {count_body}")
    print("END Summary create_files_from_excel_columns()>>")
###############################################################################


def get_unique_records(local_db_path, citations):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT citation FROM Results")
        citations_db = [row[0] for row in c.fetchall()]
        unique_citations = [x for x in citations if x not in citations_db]
    return unique_citations
###############################################################################


def extract_text_form_pdf(xml_dir, local_db_name, local_pdf_dir):    
    ## Check if the local database exists; if not, create it
    local_db_name = (local_db_name + "_local.db").lower()
    local_DB_dir = "Local_DB"
    os.makedirs(local_DB_dir, exist_ok=True)
    local_db_path = "../" + local_DB_dir + "/" + local_db_name
    
    if not os.path.isfile(local_db_path):
        try:
            with sqlite3.connect(local_db_path) as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS Results
                                    (result_id INTEGER PRIMARY KEY, 
                                    body TEXT, 
                                    citation TEXT, 
                                    processed INTEGER DEFAULT 0)''')
                conn.commit()
        except Exception as e:
            return {"code" : "failure", "msg" : f"DB cannot be created! Actual exception is: {e}"}
                
    ## Define the namespace
    namespaces = {'ns': 'http://www.tei-c.org/ns/1.0'}
        
    ## Get a list of xml files and extract pmids from file names
    xml_files = glob.glob(xml_dir + "/*.tei.xml")
    
    ## Create a list of headers upon hitting which code must stop extracting
    stop_headers = ['acknowledgement', 'references', 'funding', 'availability']
    full_text = []
    citation_text = []
    for xml_file in xml_files: 
        temp_xml_file = os.path.basename(xml_file)
        tree = ET.parse(xml_dir + '/' + temp_xml_file)

        ## Build a dictionary to map children to their parent
        parent_map = {c: p for p in tree.iter() for c in p}
        
        ## Find all paragraph elements
        paragraphs = tree.findall('.//ns:p', namespaces)
        
        ## Extract and print the text from each paragraph that's not inside an 'acknowledgement' div
        all_text = ""
        abstract_text = ""
        
        stop_flag = False
        
        for p in paragraphs:
            parent = parent_map.get(p)
            skip = False
            while parent is not None:
                if parent.tag == '{http://www.tei-c.org/ns/1.0}profileDesc':
                    for abstract in parent.findall('.//ns:abstract', namespaces):
                        if abstract is not None:
                            abstract_text += p.text + "\n"  # Add abstract text
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
                text = ''.join(pmcd.iter_paragraphs(p))
                if len(text.split()) < 10:
                    continue
                all_text += text + "\n"
        all_text = abstract_text + all_text
        all_text = all_text.replace("\n", " ").lower()        
        all_text = pmcd.remove_newline_multiple_spaces(pmcd.remove_unwanted_spaces(all_text))
        if all_text:
            full_text.append(all_text)

            pdf_file = local_pdf_dir + "/" \
                        + os.path.basename(xml_file).rstrip(".tei.xml") + ".pdf"
            citation_text.append(pdf_file)

    df = pd.DataFrame({'body' : full_text,
                       'citation' : citation_text,
                       'processed' : [0]*len(full_text)
        })
    return {"df" : df, "local_db_path" : local_db_path}
###############################################################################
local_db_name = input("Enter a name for local DB, NO spaces BUT you can use underscore(s) = ")

if len(local_db_name) == 0:
    local_db_name = "pdf_namespace"
    
xml_dir = pdf_dir + "_xml"
results = extract_text_form_pdf(xml_dir, local_db_name, pdf_dir)

df = results["df"]
local_db_path = results["local_db_path"]
table_name = "Results"

## Push only unique records in local DB
uniqe_records = get_unique_records(local_db_path, df.citation)
if bool(uniqe_records):
    with sqlite3.connect(local_db_path) as conn:
        c = conn.cursor()
    
        # Convert the DataFrame to a list of tuples
        results = [tuple(x) for x in df.values]
        
        # add the search_id to each result
        c.executemany('''INSERT INTO Results(body, citation, processed) 
                         VALUES(?, ?, ?)''', results)
        conn.commit()
else:
    print("No unique records are found, nothing will be pushed")
    
    
##-------------------- Extract text from xml into DB ------------------------##
# Read the data from local db to a df
data_dir="../tmp"
df = pmcd.fetch_all_data_as_df(local_db_path, table_name)

# Only extract those rows from df that are not yet processed
df = df[df.processed == 0]

# Now create files from the columns of this dfcd 
if df.empty:
    msg = "There is no new record to process, execution is terminated"
    print(msg)
else:
    try:
        pmcd.create_tmp_dir(data_dir)
    except:
        msg = f"{os.path.abspath(data_dir)} cannot be created!"
        print(msg)
    try:
        create_files_from_excel_columns(data_dir, df)
    except:
        msg = f"Could not create files in {os.path.abspath(data_dir)}"
        print(msg)

# load results data
chunk_size = 2000
loader = DirectoryLoader(data_dir + '/', glob='**/body*.txt')
results = loader.load()
print(f'You have {len(results)} body texts')

print(f'Spliliting long body texts into smaller chunks of size = {chunk_size}')

## Pre-process the body text of each record
res_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=0)

result_chunks_indexed = []

# split the text in result files into managable size chunks for openai api calls
for result in results:
    result.page_content = pmcd.remove_newline_multiple_spaces(result.page_content)
    results_splitted = res_splitter.split_documents([result])

    for chunk_idx, chunk in enumerate(results_splitted):
        chunk.metadata['chunk_id'] = chunk_idx
    result_chunks_indexed.append(results_splitted)

## From list of lists to list
result_chunks_indexed = [item for sublist in result_chunks_indexed for item in sublist]
    
print(f'After splitting body into smaller texts you have {len(result_chunks_indexed)} texts')

# load citation data
loader = DirectoryLoader(data_dir + '/', glob='**/citation*.txt')
citations = loader.load()

# sort the results and citations to achive one-to-one correspondence
result_chunks_indexed = pmcd.sort_list(result_chunks_indexed)
citations = pmcd.sort_list(citations)

all_texts = result_chunks_indexed
print(f'you have {len(all_texts)} texts')

# extract only part of filename from citation metadata
pattern = r"\d+\.txt"
citation_list = [record.metadata for record in citations]
number_txt_list = [re.search(pattern, item['source']).group(0)
                    for item in citation_list]

all_texts_citated = []
for record in all_texts:
    matching_citation = pmcd.get_matching_meta(record, citations, number_txt_list)
    record.metadata["citation"] = matching_citation['matching_citation']
    record.metadata["paper_id"] = matching_citation['paper_id']
    all_texts_citated.append(record)
    
##---------------------- Now push vectors into pinecone ---------------------##
# pdb.set_trace()
namespace = local_db_name
# embedd_model = "biobert"
embedd_model = "MedCPT"
try:
    pmcd.push_vectors_into_pinecone(all_texts_citated,
                                    namespace,
                                    embedd_model)
            
    pmcd.update_all_rows(local_db_path,
                        table_name,
                        column_name="processed",
                        new_value=1)
        
    msg = f"Vectors pushed successfully into pinecone DB"
    print(msg)
except Exception as e:
    msg = f"Error pushing the vectors in to pinecone, actual exception is: {e}"
    print(msg)
