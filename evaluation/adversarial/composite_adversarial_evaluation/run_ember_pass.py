import numpy as np

import sys, os
repo_root = "/data/quo.vadis/"
sys.path.append(repo_root)
from utils.hashpath import get_rawpe_db

from secml_malware.models.malconv import MalConv
from secml.array import CArray
from secml_malware.models.c_classifier_ember import CClassifierEmber
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi

adversarial_testset_path = repo_root + "data/adversarial.emulation.dataset/reports_ember"
adversarial_testset_files = os.listdir(adversarial_testset_path)
adversarial_reports = [x.rstrip(".json") for x in adversarial_testset_files if x.endswith(".json")]

db_orig = get_rawpe_db()
db_adv_malconv = get_rawpe_db(PE_DB_PATH="/data/quo.vadis/adversarial/samples_adversarial_testset_gamma_ember")

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
    path = db_adv_malconv[h]
    y_malconv_adv.append(get_y(net, path))

np.save("y-gamma-vs-ember-scores-orig.npy", np.stack(y_malconv_orig))
np.save("y-gamma-vs-ember-scores.npy", np.stack(y_malconv_adv))