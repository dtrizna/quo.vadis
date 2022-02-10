import speakeasy
import pandas as pd
import numpy as np
import os
import time
import pickle

MALWARE_PATH = "../../../data/pe.dataset/"
X86_PATH = MALWARE_PATH + "PeX86Exe/"
X86_RANSOMWARE = X86_PATH + "ransomware/"
X86_CLEAN = X86_PATH + "clean/"

FOLDER = X86_CLEAN

import pdb; pdb.set_trace()

from pefile import PEFormatError
from unicorn import UcError

start_idx = 0
end_idx = 50
files = [x for x in os.listdir(FOLDER)[start_idx:end_idx]]
reports = {}
timedeltas = []

now = time.time()
for i, file in enumerate(files):
    print(f"\n[*] {i}/{len(files)} : {FOLDER+file}")
    se = speakeasy.Speakeasy()
    try:
        module = se.load_module(FOLDER+file)
        se.run_module(module)
        
        report = se.get_report()
        reports[file] = report
        
        # reporting during execution
        aa = pd.json_normalize(report)
        entry_points = pd.json_normalize(aa["entry_points"].iloc[0])
        print(f"Len of entry_points: {entry_points.shape[0]}")
        print(f"\t* api seq. lengths: {[len(x['apis']) for _,x in entry_points.iterrows()][0:4]}")
        if "error.type" in entry_points.keys():
            print(f"\t* errors: ", entry_points["error.type"].values[0:4])
            for error_type in entry_points["error.type"].values:
                for _, row in entry_points[entry_points["error.type"] == error_type].iterrows():
                    if error_type == "unsupported_api":
                        print("\t\t", error_type, row["error.api_name"])
                    if error_type in ["invalid_read", "invalid_fetch"]:
                        print("\t\t", error_type,": ", row["error.instr"])
                    if error_type == "Invalid memory write (UC_ERR_WRITE_UNMAPPED)":
                        print("\t\t", error_type, ": \n", row["error.traceback"])
        # calculations
        took = time.time()-now
        timedeltas.append(took)
        now = time.time()
        print(f"[!] Took: {took:.2f}s\n")

    except PEFormatError as ex:
        print(f"\nfailed {file}", ex)
    except UcError as ex:
        print(f"\nfailed {file}", ex)

print(f"\naverage analysis time per sample: {np.mean(timedeltas)}")

with open(f"reports_{'_'.join(FOLDER.split('/')[-3:-1])}_{start_idx}_{end_idx}", "wb") as fhandle:
    pickle.dump(reports, fhandle, protocol=pickle.HIGHEST_PROTOCOL)
