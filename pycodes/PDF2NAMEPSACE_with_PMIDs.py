#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:31:25 2024

@author: wasim
"""
#!/usr/bin/bash
# make sure to activate proper conda environment by running `mamba activate qa`
import time
import subprocess
import PMC_downloader_Utils as  pmcd
import selenium_pdf_download as selen
from tkinter import filedialog as fd
import sys
import pdb
import os
from selenium.webdriver.firefox.options import Options
import glob
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

###############################################################################
##------------------------------ PDF 2 xml ----------------------------------##
email = os.getenv("EMAIL")

### Selenium pdf download
file = fd.askopenfile(title = "Select a sqlite DB file with .db extension")
if bool(file):
    if file.name.endswith(".db"):
        local_db_path = file.name
        print(f"Selected file is: {local_db_path}")
    else:
        sys.exit("Selected file does not have .db extension, most likely NOT a sqlite DB")
else:
    sys.exit("No file selected")
    
db_table = "Results"
download_dir = "../pdf_downloaded"
ncbi_url = "https://www.ncbi.nlm.nih.gov/"

if not os.path.exists(download_dir):
    os.mkdir(download_dir)

## Set selenium options
options = Options()
options.add_argument("--headless") # New option to run Firefox in headless mode
options.set_preference('permissions.default.stylesheet', 2)
options.set_preference('permissions.default.image', 2)
options.set_preference('browser.download.folderList', 2)
options.set_preference('browser.download.dir', os.path.abspath(download_dir))
options.set_preference('browser.download.manager.showWhenStarting', False)
options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/pdf')
options.set_preference('pdfjs.disabled', True) 


df = pmcd.fetch_all_data_as_df(local_db_path, db_table)
idx_body_missing = list(df[df.body == ""].index)
df = df.iloc[idx_body_missing,:]
pdb.set_trace()
pmids_pmcids_dict = pmcd.pmid2pmcid(email = email, 
                                    pmids = list(map(str, list(df.pmid))))

for pmid, pmcid in pmids_pmcids_dict.items(): 
    if pmcid != "-":
        selen.download_pmc_pdf(pmid, pmcid, ncbi_url, options, download_dir)
        time.sleep(5)                
###############################################################################
# Start GROBID server as a background process
subprocess.Popen('bash -i -c grobid-run', shell=True)

# Wait for a few seconds to ensure the server is up and running
time.sleep(10)  # Adjust this based on how long your server typically takes to start

# Now continue with the client
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="../grobid_client_python-master/config.json")

xml_dir = download_dir + "_xml"
client.process("processFulltextDocument", 
                input_path=download_dir, 
                output=xml_dir,
                n=10,  # may want to adjust this based on the cores available
                force=True)

### extract text  from XML and update teh DB table
df2 = pmcd.extract_text_form_pdf(xml_dir, local_db_path)
pmcd.replace_table_contents(local_db_path, db_table, df2)

##-------- Extract text from xml and push vectors into pinecone -------------##
namespace = input("Enter a name for namesspace, NO spaces BUT you can use underscore(s) = ")
if len(namespace) == 0:
    namespace = "pdf_namespace_test"

embedd_model = "biobert"

pmcd.preprocess_data_qa(local_db_path, 
                       db_table, 
                       namespace, 
                       embedd_model, 
                       chunk_size=2000, 
                       data_dir="../tmp")