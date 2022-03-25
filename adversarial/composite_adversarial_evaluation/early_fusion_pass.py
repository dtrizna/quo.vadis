import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np

import sys
import time
repo_root = "../../"
sys.path.append(repo_root)
from models import CompositeClassifier

adversarial_testset_path = repo_root + "data/adversarial.emulation.dataset/reports_ember"
adversarial_testset_files = os.listdir(adversarial_testset_path)
adversarial_reports = [x.rstrip(".json") for x in adversarial_testset_files if x.endswith(".json")]
print("Successfull adversarial emulation reports: ", len(adversarial_reports))
adversarial_errors = [x for x in adversarial_testset_files if x.endswith(".err")]
print("Errored adversarial emulation reports: ", len(adversarial_errors))
print("Total adversarial samples: ", len(adversarial_errors) + len(adversarial_reports))
print(f"Emulation success rate: {len(adversarial_reports)/(len(adversarial_reports)+len(adversarial_errors))*100:.2f}%")


classifier = CompositeClassifier(late_fusion_model="LogisticRegression", repo_root=repo_root,
                                emulation_report_path="data/adversarial.emulation.dataset/reports_ember",
                                rawpe_db_path="adversarial/samples_adversarial_testset_gamma_ember")
# preprocess adversarial samples using correct report path
x_adv_ember = classifier.preprocess_hashlist(adversarial_reports, dump_xy=True) # takes ~30m
np.save(repo_root+"adversarial/composite_adversarial_evaluation/X-gamma-vs-ember-early-fusion-pass.arr", x_adv_ember)

classifier = CompositeClassifier(late_fusion_model="LogisticRegression", repo_root=repo_root)
# preprocess adversarial samples using correct report path
x_orig_ember = classifier.preprocess_hashlist(adversarial_reports, dump_xy=True) # takes ~30m
np.save(repo_root+"adversarial/composite_adversarial_evaluation/X-gamma-vs-ember-early-fusion-pass-orig.arr", x_orig_ember)