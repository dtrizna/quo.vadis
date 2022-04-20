import numpy as np

import sys, os
repo_root = "/data/quo.vadis/"
sys.path.append(repo_root)
from utils.structures import rawpe_db

#from secml_malware.models.malconv import MalConv
from secml.array import CArray
from secml_malware.models.c_classifier_ember import CClassifierEmber
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi

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

adversarial_emulated_files = os.listdir(repo_root + ADVERSARIAL_EMULATED_SET_FOLDER)
adversarial_reports = [x.replace(".json","") for x in adversarial_emulated_files if x.endswith(".json")]
ADV_REPORTS = [x for x in adversarial_reports if x in ADV_SAMPLE_HASHES]

db_orig = rawpe_db()
db_ember = rawpe_db(PE_DB_PATH=repo_root + ADVERSARIAL_RAW_SET_FOLDER)

net = CClassifierEmber(tree_path=repo_root+"modules/sota/ember/parameters/ember_model.txt")
net = CEmberWrapperPhi(net)

def get_y(net, path):
    with open(path, "rb") as fhandle:
        code = fhandle.read()        
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    
    _, confidence = net.predict(x, True)
    y = confidence[1][0].item()
    return y

y_ember_orig = []
y_ember_adv = []
l = len(ADV_REPORTS)
for i,h in enumerate(ADV_REPORTS):
    print(f"{i}/{l}", end="\r")
    # orig:
    path = db_orig[h]
    y_ember_orig.append(get_y(net, path))
    # adv
    path = db_ember[h]
    y_ember_adv.append(get_y(net, path))

np.save(ARRAY_FOLDER + "y-gamma-vs-ember-scores-orig-only-adv.npy", np.stack(y_ember_orig))
np.save(ARRAY_FOLDER + "y-gamma-vs-ember-scores-only-adv.npy", np.stack(y_ember_adv))