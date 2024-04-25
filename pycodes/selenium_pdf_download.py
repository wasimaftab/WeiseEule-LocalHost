import pdb
import os
import time
import csv
import pandas as pd
import copy

from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ncbi_url = "https://www.ncbi.nlm.nih.gov/"

# download_dir = "pdf_downloaded"

# if not os.path.exists(download_dir):
#     os.mkdir(download_dir)
    
# options = Options()
# options.add_argument("--headless") # New option to run Firefox in headless mode
# options.set_preference('permissions.default.stylesheet', 2)
# options.set_preference('permissions.default.image', 2)
# options.set_preference('browser.download.folderList', 2)
# options.set_preference('browser.download.dir', os.path.abspath(download_dir))
# options.set_preference('browser.download.manager.showWhenStarting', False)
# options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/pdf')
# options.set_preference('pdfjs.disabled', True)
###############################################################################
## Function to get the most recently downloaded file in a directory
def get_latest_downloaded_file(download_dir):
    # Get list of files in the directory, filtered by modification time
    files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    # Get the most recent file
    latest_file = max(files, key=os.path.getmtime)
    # return os.path.basename(latest_file)
    return latest_file
###############################################################################

def download_pmc_pdf(pmid, pmcid, ncbi_url, options, download_dir):
    browser = webdriver.Firefox(options=options)
    pmc_url = ncbi_url + "pmc/articles/" + pmcid + "/"
    browser.get(pmc_url)
    try:
        wait = WebDriverWait(browser, 20)
        pdf_links = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.pmc-sidebar__formats li a')))

        # Iterate over all links and navigate to the link containing '.pdf' in the href attribute
        for link in pdf_links:
            href = link.get_attribute('href')
            if '.pdf' in href:
                # Open the pdf link in a new window
                browser.execute_script(f'window.open("{href}","_blank");')
                break

        time.sleep(25)  # delay for 10 seconds
        # After the file is downloaded rename it by PMID:
        ## rename downloaded pdf
        downloaded_file_path = get_latest_downloaded_file(download_dir)
        if downloaded_file_path:
            new_file_path = download_dir + "/" + pmid + ".pdf"
            os.rename(downloaded_file_path, new_file_path)
                
                
        # with open(f'{download_dir}/file_mapping.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([pmid, os.path.basename(href)])
            
        ## Switch to the new window and close it 
        ## (this is a hack to deal with bowser freezing upon downloading)
        browser.switch_to.window(browser.window_handles[-1])
        browser.close()

    except Exception as e:
        print(e)
    finally:
        browser.quit()  # quit the browser outside the loop