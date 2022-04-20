import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np

import sys
import time
repo_root = "/data/quo.vadis/"
sys.path.append(repo_root)
from models import CompositeClassifier

def get_adversarial_samples(folder):
    fullpaths = [repo_root+folder+x for x in os.listdir(repo_root+folder)]
    adversarial_samples = [x for x in fullpaths if not os.path.islink(x)]
    adversarial_samples.sort()
    return adversarial_samples

# ADVERSARIAL_EMULATED_SET_FOLDER = "data/adversarial.emulated/reports_ember_15sections_10population/"
#ADVERSARIAL_EMULATED_SET_FOLDER = "data/adversarial.emulated/reports_ember_10sections_10population/"
ADVERSARIAL_EMULATED_SET_FOLDER = "data/adversarial.emulated/reports_ember_5sections_10population/"

#ADVERSARIAL_RAW_SET_FOLDER = "data/adversarial.samples/samples_adversarial_testset_gamma_ember_15sections_10population/"
#ADVERSARIAL_RAW_SET_FOLDER = "data/adversarial.samples/samples_adversarial_testset_gamma_ember_sections/10/"
ADVERSARIAL_RAW_SET_FOLDER = "data/adversarial.samples/samples_adversarial_testset_gamma_ember_sections/5/"

ADV_SAMPLES = get_adversarial_samples(ADVERSARIAL_RAW_SET_FOLDER)
ADV_SAMPLE_HASHES = [x.split("/")[-1] for x in ADV_SAMPLES]

#ARRAY_FOLDER  = "evaluation/adversarial/composite_adversarial_evaluation/arrays_ember_15sections_10population/"
#ARRAY_FOLDER  = repo_root + "evaluation/adversarial/composite_adversarial_evaluation/arrays_ember_10sections_10population/"
ARRAY_FOLDER  = repo_root + "evaluation/adversarial/composite_adversarial_evaluation/arrays_ember_5sections_10population/"
os.makedirs(ARRAY_FOLDER, exist_ok=True)

# ====================================
# STATISTICS
adversarial_emulated_files = os.listdir(repo_root + ADVERSARIAL_EMULATED_SET_FOLDER)
adversarial_reports = [x.replace(".json","") for x in adversarial_emulated_files if x.endswith(".json")]
ADV_REPORTS = [x for x in adversarial_reports if x in ADV_SAMPLE_HASHES]

adv_only_reports = len(ADV_REPORTS)
print("Successful adversarial emulation reports: ", adv_only_reports)

adversarial_errors = len(ADV_SAMPLE_HASHES) - adv_only_reports
print("Errored adversarial emulation reports: ", adversarial_errors)

print("Total adversarial samples: ", adversarial_errors + adv_only_reports)
print(f"Emulation success rate: {adv_only_reports/(adv_only_reports+adversarial_errors)*100:.2f}%")

# ====================================
# EARLY FUSION PASSES
# 1. ADVERSARIAL SAMPLES
classifier = CompositeClassifier(root=repo_root, 
                                emulation_report_path=ADVERSARIAL_EMULATED_SET_FOLDER,
                                rawpe_db_path=ADVERSARIAL_RAW_SET_FOLDER)
# preprocess samples using correct report path
x_adv_ember = classifier.preprocess_pelist(ADV_REPORTS, dump_xy=True) # takes ~30m
np.save(ARRAY_FOLDER + "X-gamma-vs-ember-early-fusion-pass-only-adv.arr", x_adv_ember)

# 2. ORIGINAL SAMPLES - TO HAVE THE SAME HASHES (CANNOT BE ACQUIRES FROM EXISTING ARRAYS)
classifier_orig = CompositeClassifier(root=repo_root)
# preprocess samples using correct report path
x_orig_ember = classifier_orig.preprocess_pelist(ADV_REPORTS, dump_xy=True) # takes ~30m
np.save(ARRAY_FOLDER + "X-gamma-vs-ember-early-fusion-pass-orig-only-adv.arr", x_orig_ember)

import pdb;pdb.set_trace()