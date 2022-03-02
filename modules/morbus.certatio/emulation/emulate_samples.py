import argparse
import logging

import speakeasy
from pefile import PEFormatError
from unicorn import UcError

import os
import sys
import time
import json
from numpy import sum

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collects tracelogs from sample emulation.")

    # data specifics
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--samples", type=str, nargs="+", help="Path to PE file(s) for emulation. Can specify multiple (separate by space).")
    group.add_argument("--sample-prefix", type=str, nargs="+", help="Prefix of path PE files for emulation. Can specify multiple (separate by space).")

    parser.add_argument("--output", type=str, default="reports", help="Directory where to store emulation reports.")

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
    
    logging.warning(f" [!] Starting emulation.")

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
        logging.debug(f" [*] {i}/{l}: emulating {file} ...")
        
        samplename = file.split("/")[-1]
        reportfile = f"{report_folder}/{samplename}.json"
        if os.path.exists(reportfile):
            logging.warning(f" [!] Exists, skipping analysis: {reportfile}\n")
            continue
        try:
            se = speakeasy.Speakeasy(config=json.load(open("./speakeasy_config.json")))
            module = se.load_module(file)
            se.run_module(module)
            report = se.get_report()

            took = report["emulation_total_runtime"]
            api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]])
            
            if api_seq_len > 1:
                # 2 API calls are sometimes already enough
                with open(f"{reportfile}", "w") as f:
                    json.dump(report["entry_points"], f, indent=4)
            elif api_seq_len == 1 or api_seq_len == 0:
                # some uninformative failures with 0 or 1 API calls - e.g. ordinal_100
                api = [x['apis'] for x in report['entry_points']][0] if api_seq_len ==1 else ''
                err = [x['error']['type'] for x in report['entry_points']]
                if "unsupported_api" in err:
                    err.extend([x['error']['api']['name'] for x in report['entry_points']])
                logging.debug(f" [DBG]: API nr.: {api_seq_len}; Err: {err}; APIs: {api}")

            logging.debug(f" [+] Finished emulation, took: {took:.2f}s, API calls acquired: {api_seq_len}\n")


        except PEFormatError as ex:
            logging.error(f" [-] Failed emulation, PEFormatError: {file}\n{ex}\n")
        except UcError as ex:
            logging.error(f" [-] Failed emulation, UcError: {file}\n{ex}\n")
        except IndexError as ex:
            logging.error(f" [-] Failed emulation, IndexError: {file}\n{ex}\n")
