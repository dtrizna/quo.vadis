import time
import logging
from pandas import Series, read_csv

def load_ti_csv(csv="", 
             type_whitelist=["AvastHeurStreamback", "IdpFileExtractor", "Phoenix"]):
    """Loads and cleans the filepath data (according to currently provided format).

    Args:
        csv (str, optional): CSV filename with filepaths. Defaults to "bohacek_20211022113102.csv".
        type_whitelist (list, optional): List with type to preserve. Defaults to ["AvastHeurStreamback", "IdpFileExtractor", "Phoenix"]
    Returns:
        pd.Series(): Series object with only relevant filepaths.
    """
    if not csv:
        return Series()

    df = read_csv(csv, sep=";", 
            usecols=["file_name", "Type", "Detections [avast9]"],
            dtype={"file_name": "object", "Type": "category"})#, "Detections [avast9]": "category"})

    # Type cleanup
    if type_whitelist:
        df = df[df.Type.apply(lambda x: x in type_whitelist)]

    # Rename and leave only relevant data
    df = df[["file_name", "Detections [avast9]"]]
    df.columns = ["x","y"]

    # Content cleanup based on filepath
    df = df.drop(df[df.x.isna()].index) # NaN
    df = df.drop(df[df.x.str.contains("http[s]?", regex=True)].index) # URLs
    df = df.drop(df[df.x.str.contains("pro/malicious_scripts")].index) # Avast TI proprietary, noise
    df = df.drop(df[df.x.str.split("\\").apply(lambda x: len(x) <= 1)].index) # only filenames, no paths
    df = df.drop(df[df.x.str.contains(r"idp\\[a-f0-9]", regex=True)].index) # Avast IDP memory scans
    df = df.drop(df[df.x.str.contains(r"<subj[^>]+>", regex=True)].index) # E-mail data, no paths

    # setting binary $y \in (0,1)$
    df.y = df.y.fillna(0)
    df.loc[df.y != 0, "y"] = 1

    df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    
    path_dataset_folder = "/data/quo.vadis/data/path.dataset/"
    ti_csv = path_dataset_folder + "bohacek_20211022113102.csv"

    labio_csv_folder = path_dataset_folder + "labio_csv/"


    logging.warning(f" [*] {time.ctime()}: Loading data from a CSV file: {ti_csv} ")
    df = load_ti_csv(ti_csv)

    malicious_ti = [x.replace(",","")+"\n" for x in df[df.y == 1].x.values]
    benign_ti = [x.replace(",","")+"\n" for x in df[df.y == 0].x.values]

    with open(path_dataset_folder + "dataset_malicious_ti.txt", "w") as f:
        f.writelines(malicious_ti)
    
    with open(path_dataset_folder + "dataset_benign_ti.txt", "w") as f:
        f.writelines(benign_ti)

    logging.warning(f" [*] {time.ctime()}: Parsing of TI CSV file finished...")

    
    
