import requests, json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from vyvar.oauth2 import OAuth2Token

import sys, os

TOKENFILE = ".theHive.keycloak.secret.api"
LABIO_API_ROOT = ""
KEYCLOAK_URL = ""

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

def get_filename_from_hash(hhash):
    search = f"sha256: {hhash}"
    query = {
        "limit": 1,
        "offset": 0,
        "cols": ["sources", "sha256", "tagger"],
        "query": search
        }

    r = labio_api(f"samples/search", TOKENFILE, body=json.dumps(query))
    filename = "C:\\Users\\user\\Downloads\\exploit.exe"
    try:
        items = json.loads(r.content.decode())['items']
    except KeyError:
        print(f"[-] Weird response for {hhash}. Setting filename to: {filename}")
        return filename
    try:
        filename = [x for x in items[0]["sources"]["props"]["file_name"] if "\\" in x][0]
    except IndexError:
        # relative path only, place it in downloads folder
        filename = "C:\\Users\\user\\Downloads\\" + items[0]["sources"]["props"]["file_name"][0]

    return filename


if __name__ == "__main__":
    folder = sys.argv[1]
    files = os.listdir(folder)
    l = len(files)
    
    dbname = folder+"sha_path_db.csv"
    dbhandle = open(dbname, "a")

    print(f"[!] Hashfile folder: {folder} | Total files: {l}")
    print(f"[!] Filepath DB: {dbname}")
    print(f"[*] Starting enumeration of filepaths from Lab.IO...")
    for i,file in enumerate(files):
        print(f"{i}/{l}", end="\r")
        
        hhash = file.replace(".dat","")
        filename = get_filename_from_hash(hhash)
        if not filename:
            print(f"[-] {hhash} doesn't have a filename...")
        dbhandle.write(f'{hhash},"{filename}"\n')

    dbhandle.close()
        