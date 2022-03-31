import os
import pandas as pd

def filepath_db(FILEPATH_CSV_LOCATION="/data/quo.vadis/data/pe.dataset/PeX86Exe"):
    """ 
    Reads CSV files in specified path. First column: PE identifier (e.g. hash), second column: PE path on target system, e.g. 
    bd4efcbd4fd664f4ddb9be03f42b2ca98d5f28cfbe24024a93a4798f5a0b15a0,C:\\Users\\Andy\\AppData\\Roaming\\Windows Update.exe
    """
    hashpath_DB = pd.DataFrame()
    if not FILEPATH_CSV_LOCATION:
        return {}
    for root, dirs, _ in os.walk(FILEPATH_CSV_LOCATION):
        for name in dirs:
            fullpath = os.path.join(root, name)
            csvdb = os.path.join(fullpath, [x for x in os.listdir(fullpath) if x.endswith(".csv")][0])
            df = pd.read_csv(csvdb, header=None)
            hashpath_DB = pd.concat([hashpath_DB,df])
    return dict(hashpath_DB.to_dict("split")["data"])


def report_db(REPORT_PATH="/data/quo.vadis/data/emulation.dataset"):
    db = {}
    if not REPORT_PATH:
        return db
    for root, dirs, _ in os.walk(REPORT_PATH):
        dirs = [x for x in dirs if "report_" in x]
        for name in dirs+[root]:
            if name != root:
                fullpath = os.path.join(root, name)
                files = os.listdir(fullpath)
            else:
                fullpath = root
                files = set(os.listdir(root)) - set(dirs)
            reportlist = [x for x in files if x.endswith(".json")]
            for h in reportlist:
                db[h.rstrip(".json")] = os.path.join(fullpath, h)
    return db


def rawpe_db(PE_DB_PATH="/data/quo.vadis/data/pe.dataset/PeX86Exe"):
    db = {}
    if not PE_DB_PATH:
        return db
    for root, dirs, _ in os.walk(PE_DB_PATH):
        for name in dirs+[root]:
            if name != root:
                fullpath = os.path.join(root, name)
                files = os.listdir(fullpath)
            else:
                fullpath = root
                files = set(os.listdir(root)) - set(dirs)
            for pehash in files:
                db[pehash] = os.path.join(fullpath, pehash)
    return db