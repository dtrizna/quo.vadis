import os
import pandas as pd

def get_hashpath_db(PE_DB_PATH="/data/quo.vadis/data/pe.dataset/PeX86Exe"):
    hashpath_DB = pd.DataFrame()
    for root, dirs, _ in os.walk(PE_DB_PATH):
        for name in dirs:
            fullpath = os.path.join(root, name)
            csvdb = os.path.join(fullpath, [x for x in os.listdir(fullpath) if x.endswith(".csv")][0])
            df = pd.read_csv(csvdb, header=None, names=["hash", "path"])
            hashpath_DB = hashpath_DB.append(df)
    return hashpath_DB


def get_path_from_hash(h, db):
    return db[db["hash"] == h].path.values[0]