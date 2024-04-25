#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:28:26 2024

@author: wasim
"""
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import pandas as pd
import time
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import random

options = Options()
options.add_argument("--headless")

# Set the options when initializing the driver
driver = webdriver.Firefox(options=options)

# Navigate to Google's homepage
driver.get("https://www.google.com")

def search_google(search_query):
    try:
        # Wait until the cookie consent form is loaded and then accept it
        wait = WebDriverWait(driver, 10)
        try:
            consent_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="L2AGLb"]')))
            consent_button.click()
        except (NoSuchElementException, TimeoutException):
            print("No consent button found, or it has already been accepted.")
        
        # Find the search bar element
        try:
            search_box = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
            search_box.clear()  # Clear any pre-filled text in the search box
            search_box.send_keys(search_query + Keys.RETURN)
        except (NoSuchElementException, TimeoutException):
            print("Search box not found.")
            return None
        
        # Wait for the search results to load
        time.sleep(5)
        search_results = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".tF2Cxc")))
        for result in search_results[:5]:
            title_element = result.find_element(By.CSS_SELECTOR, ".DKV0Md")
            title = title_element.text
            link_element = result.find_element(By.CSS_SELECTOR, ".yuRUbf a")
            link = link_element.get_attribute("href")
            if "pubmed" in link:
                return link
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
###############################################################################   
    
df = pd.read_excel("~/Downloads/My_GPT/Dosage_comp_zizis.xlsx")
# df = df[:5]
link = []
pmid = []
start = time.time()
flag = 0
for index, row in df.iterrows():
    if pd.isna(row.iloc[1]):
        pmid.append(np.NAN)
        link.append(np.NAN)
    else:
        print(f"====Google search for {row.iloc[1]}====")
        url = search_google(row.iloc[1])
        if url is not None:
            print(f"====PubMed link {url}====\n\n")
            link.append(url)
            pmid.append(url.rstrip('/').split('/')[-1])
        else:
            link.append("No link found")
            pmid.append("No PMID found")

    # Sleep for some seconds for Google to not block you
    time.sleep(random.randint(10,20))
    
# Close the browser outside the loop, after all iterations are done
driver.quit()

end = time.time()
print(f"Elapsed = {end - start}")
