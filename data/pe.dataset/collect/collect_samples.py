from vyvar.oauth2 import OAuth2Token
import requests
import json
import os
import time
from pandas import Series
import sys
import subprocess
from hashlib import sha256

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LABIO_API_ROOT = "https://labio-search.svc.int.avast.com/v1/"
SAMPLE_API_ROOT = "https://samples.tlab.hcp-prod-01-prg5.int.avast.com/hs3/"
KEYCLOAK_URL = "https://keycloak-tlabs.svc.int.avast.com/auth/realms/viruslab/protocol/openid-connect/token"
TOKENFILE = "collect/.theHive.keycloak.secret.api"

def get_oauth_token(keyfile):
    with open(keyfile) as f:
        keycloak_key = f.read().strip()    
    config = {
        "token_url": KEYCLOAK_URL,
        "client_id": "thehive", 
        "client_secret": keycloak_key
    }
    oauth2 = OAuth2Token(**config)
    return oauth2.get_token()


def labio_api(url, tokenfile, body=None, proxies=None):
    """Makes an API request towards LAB.IO.

    Args:
        url (str): /v1/ API url
        token (dict): OAuth token
        body (str, optional): Make a POST request with this HTTP body. Defaults to None.
        proxies (dict, optional): Make a HTTP request through this proxy. Defaults to None.

    Returns:
        [type]: [description]
    """
    token = get_oauth_token(tokenfile)
    headers = {
        "Content-type": "application/json",
        "Authorization": f"{token['token_type']} {token['access_token']}"
    }

    if body:
        r = requests.post(LABIO_API_ROOT + url, headers=headers, data=body, proxies=proxies, verify=False)
    else:
        r = requests.get(LABIO_API_ROOT + url, headers=headers, proxies=proxies, verify=False)
    return r


def download_sample(sha, proxies=None):
    """
    Args:
        sha (str): sha256 of file to download
        proxies (dict, optional): Make a HTTP request through this proxy. Defaults to None.

    Returns:
        [boolean], [str]: 
            boolean - whether download suceeded, 
            str - contents of PE or http body of error 
    """
    sha_download_path = f"{sha[0:2]}/{sha[2:4]}/{sha[4:6]}/{sha}"
    r = requests.get(SAMPLE_API_ROOT + sha_download_path, proxies=proxies, verify=False)
    if r.status_code == 200:
        return True, r.content
    else:
        return False, r.content


def dump_db(series, filepath):
    series.index.name = "sha256"
    series.name = "filepath"
    series.to_csv(filepath)


if __name__ == "__main__":

    filetype = "PeX86Exe" # "Pex64Exe" # 
    os.makedirs(filetype, exist_ok=True)
    
    q = sys.argv[1]
    searchtype = q.split(":")[1]
    os.makedirs(filetype+"/"+searchtype, exist_ok=True)

    offset = 0
    query_size = 500
    db = Series(dtype="object")

    try:
        while True:
            labio_query = f"srcprop.fileName: \"c:*\" and filetype: {filetype} and tagger.confidence: >= 50 and {q}" # severity:clean or type:ransomware
            query = {
                "limit": query_size,
                "offset": offset,
                "cols": ["sources", "sha256", "tagger"],

                # srcprop.fileName: \"c:*\" : this takes long reponse from back-end, ~ 1 min 40 sec
                "query": labio_query
                }
            print(f"\n[*] {time.ctime()}:\nmaking search request with offset {offset}...")
            print("labio query: ", labio_query)
            offset += query_size # for next iteration

            time.sleep(300)
            try:
                response = labio_api("samples/search", TOKENFILE, body=json.dumps(query))
            except requests.exceptions.ConnectionError:
                time.sleep(300)
                response = labio_api("samples/search", TOKENFILE, body=json.dumps(query))
            print("parsing response...")
            if response.status_code == 200:
                items = json.loads(response.text)["items"]
                
                print(f"got {len(items)} samples...")
                for i,sample in enumerate(items):
                    print(f" {i}/{len(items)}", end="\r")
                    sha = sample["sha256"]

                    # get filename
                    filename = None
                    try:
                        filename = [x for x in sample["sources"]["props"]["file_name"] if "\\" in x][0]
                    except IndexError:
                        # relative path only, place it in downloads folder
                        filename = "C:\\Users\\user\\Downloads\\" + sample["sources"]["props"]["file_name"][0]
                    except KeyError:
                        print(f"no filpath: {sha}")
                        continue
                    
                    # if not in folder already (since some hashes appear twice in search) - download
                    if not os.path.exists(f"{filetype}/{searchtype}/{sha}"):
                        # store it in db - if not present on a filesystem - still can use filepath for quo.vadis
                        Series({sha: filename}).to_csv(f"{filetype}/{searchtype}/sha_path_db.csv", mode="a", header=False)
                        
                        try:
                            success, pe = download_sample(sha)
                            if success:
                                # save PE 
                                localname = f"{filetype}/{searchtype}/{sha}"
                                with open(localname, "wb") as f:
                                    f.write(pe)
                                
                                fileformat = subprocess.check_output(f"file {localname}".split()).decode()
                                if ".Net" in fileformat:
                                    # we don't need .NET since emulator fails to process them
                                    #print(".NET file: ", fileformat.strip())
                                    _ = subprocess.check_output(f"rm {localname}".split())
                                
                        except MemoryError:
                            print(f"hit memory error: {sha}")
                            pass

                        # delay between dowloads
                        time.sleep(0.5)
            else:
                print(response.status_code)
                if offset > 10000:
                    offset = 0
                else:
                    print(f"[-] failure to query lab.io API with:\n{json.dumps(query)}\nwaiting 1 hour...")
                    time.sleep(360)
            
    except KeyboardInterrupt:
        #dump_db(db, f"{type}/sha_path_db.csv")
        sys.exit(0)
