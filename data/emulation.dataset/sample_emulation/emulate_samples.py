import argparse
import logging

import os
import sys
import time
import threading

sys.path.append("/data/quo.vadis/") # repo root
from preprocessing.emulation import emulate

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collects tracelogs from sample emulation.")

    # data specifics
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--samples", type=str, nargs="+", help="Path to PE file(s) for emulation. Can specify multiple (separate by space).")
    group.add_argument("--sample-prefix", type=str, nargs="+", help="Prefix of path PE files for emulation. Can specify multiple (separate by space).")

    parser.add_argument("--output", type=str, default="reports", help="Directory where to store emulation reports.")
    
    parser.add_argument("--start-idx", type=int, default=0, help="If provided, emulation will start from file with this index.")
    parser.add_argument("--threads", type=int, default=5, help="If provided, emulation will start from file with this index.")

    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")

    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    logconfig = {
        "level": level,
        "format": "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    }
    if args.logfile:
        logconfig["handlers"] = [logging.FileHandler(args.logfile, 'a', 'utf-8')]
    logging.basicConfig(**logconfig)
    
    logging.warning(f" [!] Starting emulation.\n")

    # parsing files
    filelist = []
    if args.samples:
        for file in args.samples:
            if os.path.exists(file):
                if os.path.isdir(file):
                    logging.error(f" [-] {file} is a directory... If you want parse samples within that directory, use --sample-prefix!")
                    sys.exit(1)
                filelist.append(file)
            else:
                logging.error(f" [-] {file} does not exist...")
    elif args.sample_prefix:
        for prefix in args.sample_prefix:
            prefix = prefix.rstrip("/")
            if os.path.exists(prefix):
                # folder
                filelist.extend([f"{prefix}/{x}" for x in os.listdir(prefix)])
            else:
                folder = "/".join(prefix.split("/")[:-1])
                if os.path.exists(folder):
                    prefix = prefix.split("/")[-1]
                    files = [f"{folder}/{x}" for x in os.listdir(folder) if x.startswith(prefix)]
                    filelist.extend(files)
                else:
                    logging.error(f" [-] {folder} folder does not exist...")
    
    else:
        logging.error(" [-] Specify either --samples or --sample-prefix.")
        sys.exit(1)
    
    # emulate samples
    #timestamp = int(time.time())
    report_folder = f"{args.output}"
    os.makedirs(report_folder, exist_ok=True)

    l = len(filelist)
    
    for i,file in enumerate(filelist):    
        if i < args.start_idx:
            continue    
        while len(threading.enumerate()) > args.threads:
            time.sleep(0.1)

        t = threading.Thread(target=emulate, args=(file, report_folder, i, l))
        t.start()
        logging.debug(f" [D] Started theread: {i}\{l}")
