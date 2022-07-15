import os
import sys
import time

import pickle
import numpy as np

from ember import PEFeatureExtractor

root = "../../../"
sys.path.append(root)

def get_ember_feature_vector(file_data):
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return features

set_path = root + "data/train_val_test_sets/"
train_set = pickle.load(open(set_path+"X_train.pickle.set", "rb"))
y_train_set = np.load(set_path + "y_train.arr")
val_set = pickle.load(open(set_path+"X_val.pickle.set", "rb"))
y_val_set = np.load(set_path + "y_val.arr")

pe_trainvalset_path = root + "data/archives/pe_trainset/PeX86Exe/"

X_ember_trainset = np.empty((0,2381), dtype=np.float32)
X_ember_valset = np.empty((0,2381), dtype=np.float32)
y_ember_trainset = []
y_ember_valset = []

def get_ember_feature_vector(file_data):
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return features

set_path = root + "data/train_val_test_sets/"
train_set = pickle.load(open(set_path+"X_train.pickle.set", "rb"))
y_train_set = np.load(set_path + "y_train.arr")
val_set = pickle.load(open(set_path+"X_val.pickle.set", "rb"))
y_val_set = np.load(set_path + "y_val.arr")

pe_trainvalset_path = root + "data/archives/pe_trainset/PeX86Exe/"

X_ember_trainset = np.empty((0,2381))
X_ember_valset = np.empty((0,2381))
y_ember_trainset = []
y_ember_valset = []

malicious_folders = ["backdoor", "coinminer", "dropper", "keylogger", "ransomware", "rat", "trojan"]
for folder in malicious_folders + ["clean"]:
    hashes = os.listdir(pe_trainvalset_path + folder)
    if len(hashes) == 0:
        print(f"\nSomething is wrong with {folder}\n")
    
    fullpaths = [os.path.join(pe_trainvalset_path + folder, x) for x in hashes]

    l = len(hashes)
    print(folder, end=f" (total={l}): |")
    for i in range(l):
        print(i, end="|")
        hh = hashes[i]
        fhh = fullpaths[i]    
        with open(fhh, "rb") as f:
            bytez = f.read()
    
        feature_vector = get_ember_feature_vector(bytez)
        
        if hh in train_set:
            X_ember_trainset = np.vstack([X_ember_trainset, feature_vector])
            
            if folder in malicious_folders:
                y_ember_trainset.append(1)
            else:
                y_ember_trainset.append(0)
            
            del(train_set[train_set.index(hh)])

        elif hh in val_set:
            X_ember_valset = np.vstack([X_ember_valset, feature_vector])
            
            if folder in malicious_folders:
                y_ember_valset.append(1)
            else:
                y_ember_valset.append(0)
            
            del(val_set[val_set.index(hh)])

        else:
            print(f"\nCannot find {fhh}")

    print()


run_folder = f"output_{int(time.time())}"
os.mkdir(run_folder)

np.save(f"{run_folder}/X_ember_trainset", X_ember_trainset)
np.save(f"{run_folder}/y_ember_trainset", np.array(y_ember_trainset, dtype=int))

print(f"Left hashes in train_set: {len(train_set)}")
pickle.dump(train_set, open(f"{run_folder}/train_set.leftover.pickle", "wb"))
print(f"Dumped lefotver to: ./{run_folder}/train_set.leftover.pickle")


np.save(f"{run_folder}/X_ember_valset", X_ember_valset)
np.save(f"{run_folder}/y_ember_valset", np.array(y_ember_valset, dtype=int))

print(f"Left hashes in val_set: {len(val_set)}")
pickle.dump(val_set, open(f"{run_folder}/val_set.leftover.pickle", "wb"))
print(f"Dumped lefotver to: ./{run_folder}/val_set.leftover.pickle")