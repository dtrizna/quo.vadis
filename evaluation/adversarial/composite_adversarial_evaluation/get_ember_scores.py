import numpy as np

import sys, os
repo_root = "/data/quo.vadis/"
sys.path.append(repo_root)
from utils.structures import rawpe_db

#from secml_malware.models.malconv import MalConv
from secml.array import CArray
from secml_malware.models.c_classifier_ember import CClassifierEmber
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi

ADVERSARIAL_EMULATED_SET_FOLDER = "data/adversarial.emulated/reports_ember_10sections_10population/"
ADVERSARIAL_RAW_SET_FOLDER = "data/adversarial.samples/samples_adversarial_testset_gamma_ember_sections/10/"
ARRAY_FOLDER  = repo_root + "evaluation/adversarial/composite_adversarial_evaluation/arrays_ember_10sections_10population/"

adversarial_emulated_files = os.listdir(repo_root + ADVERSARIAL_EMULATED_SET_FOLDER)
adversarial_reports = [x.rstrip(".json") for x in adversarial_emulated_files if x.endswith(".json")]

db_orig = rawpe_db()
db_ember = rawpe_db(PE_DB_PATH=repo_root + ADVERSARIAL_RAW_SET_FOLDER)

net = CClassifierEmber(tree_path="../../modules/sota/ember/parameters/ember_model.txt")
net = CEmberWrapperPhi(net)

def get_y(net, path):
    with open(path, "rb") as fhandle:
        code = fhandle.read()        
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    
    _, confidence = net.predict(x, True)
    y = confidence[1][0].item()
    return y

y_malconv_orig = []
y_malconv_adv = []
l = len(adversarial_reports)
for i,h in enumerate(adversarial_reports):
    print(f"{i}/{l}", end="\r")
    # orig:
    path = db_orig[h]
    y_malconv_orig.append(get_y(net, path))
    # adv
    path = db_ember[h]
    y_malconv_adv.append(get_y(net, path))

np.save(ARRAY_FOLDER + "y-gamma-vs-ember-scores-orig.npy", np.stack(y_malconv_orig))
np.save(ARRAY_FOLDER + "y-gamma-vs-ember-scores.npy", np.stack(y_malconv_adv))