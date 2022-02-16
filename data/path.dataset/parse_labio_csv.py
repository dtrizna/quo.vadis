import time
import logging
from pandas import read_csv
import os
from collections import Counter

def load_plain_csv(csv=""):
    if not csv:
        return []
    df = read_csv(csv, header=None)
    paths = [x.replace(",","")+"\n" for x in df[1].values if not x.lower().endswith(".crdownload")]
    return paths


if __name__ == "__main__":
    
    path_dataset_folder = "/data/quo.vadis/data/path.dataset/"
    pe_dataset_folder = "/data/quo.vadis/data/pe.dataset/PeX86Exe/"
    labio_csv_folder = path_dataset_folder + "labio_csv/"
    csv_filename = "sha_path_db.csv"

    for folder in os.listdir(pe_dataset_folder):

        tag = folder
        csv_fullpath = pe_dataset_folder + folder + "/" + csv_filename

        logging.warning(f" [*] {time.ctime()}: Loading data from a CSV file: {csv_fullpath} ")
        paths = load_plain_csv(csv_fullpath)
        logging.warning(f" [*] {time.ctime()}: Loaded {len(paths)} lines")

        if tag + ".csv" in os.listdir(labio_csv_folder):
            file = labio_csv_folder + tag + ".csv"
            logging.warning(f" [*] {time.ctime()}: Loading data from a CSV file: {file} ")
            paths.extend(load_plain_csv(file))
            logging.warning(f" [*] {time.ctime()}: Extended to {len(paths)} lines")
        
        c = Counter(paths)
        logging.warning(f" [*] {time.ctime()}: Repeatable pattern: {c.most_common(10)}")

        paths = list(set(paths))
        logging.warning(f" [*] {time.ctime()}: Left unique paths only: {len(paths)} lines")

        if tag == "clean":
            filename = f"dataset_benign_labio.txt"
        else:
            filename = f"dataset_labio_{tag}.txt"

        with open(path_dataset_folder + filename, "w") as f:
            f.writelines(paths)

        logging.warning(f" [*] {time.ctime()}: Parsing of {tag} finished...\n")

    
    
