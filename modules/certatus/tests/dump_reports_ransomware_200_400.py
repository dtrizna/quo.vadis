import speakeasy
import pandas as pd
import numpy as np
import os
import time
import pickle

MALWARE_PATH = "../../../data/pe.dataset/"
X86_PATH = MALWARE_PATH + "PeX86Exe/"
X86_RANSOMWARE = X86_PATH + "ransomware/"

from pefile import PEFormatError
from unicorn import UcError

files_200 = [x for x in os.listdir(X86_RANSOMWARE)[200:400]]
reports = {}
timedeltas = []

now = time.time()
print("starting")
for i, file in enumerate(files_200):
    # calculations
    took = time.time()-now
    timedeltas.append(took)
    now = time.time()

    print(f"{i}/{len(files_200)}, took: {took:.2f}s")
    se = speakeasy.Speakeasy()
    try:
        module = se.load_module(X86_RANSOMWARE+file)
        se.run_module(module)
        reports[file] = se.get_report()
    except PEFormatError as ex:
        print(f"\nfailed {file}", ex)
    except UcError as ex:
        print(f"\nfailed {file}", ex)

print(f"\naverage analysis time per sample: {np.mean(timedeltas)}")

with open("reports_ransomware_200_400.pickle", "wb") as fhandle:
    pickle.dump(reports, fhandle, protocol=pickle.HIGHEST_PROTOCOL)
