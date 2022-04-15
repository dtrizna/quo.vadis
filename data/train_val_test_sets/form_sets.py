from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import pickle
import numpy as np

RANDOM_SEED = 1763

#emulation_dataset_path = "/data/quo.vadis/data/emulation.dataset/"
emulation_dataset_path = "/data/quo.vadis/data/emulation.dataset/testset_emulation/"

emulation_dataset_folders_malicious = [x for x in os.listdir(emulation_dataset_path) if "report_" in x and "clean" not in x]
print("Malicious folders: ", emulation_dataset_folders_malicious)
emulation_dataset_folders_clean = [x for x in os.listdir(emulation_dataset_path) if "report_" in x and "clean" in x]
print("Malicious folders: ", emulation_dataset_folders_clean)

X_malicious_raw = []
X_malicious_raw_error = []
for folder in emulation_dataset_folders_malicious:
    for file in os.listdir(emulation_dataset_path + folder):
        if file.endswith(".json"):
            X_malicious_raw.append(file.replace(".json","").replace(".dat","")) # .dat - bulk download naming convention
        else:
            X_malicious_raw_error.append(file.replace(".err","").replace(".dat",""))
print(f"[!] MALWARE: successfull emulations: {len(X_malicious_raw)}, errored: {len(X_malicious_raw_error)}, success ratio: {len(X_malicious_raw)*100/(len(X_malicious_raw_error)+len(X_malicious_raw)):.2f}%")

X_clean_raw = []
X_clean_raw_error = []
for folder in emulation_dataset_folders_clean:
    for file in os.listdir(emulation_dataset_path + folder):
        if file.endswith(".json"):
            X_clean_raw.append(file.replace(".json","").replace(".dat",""))
        else:
            X_clean_raw_error.append(file.replace(".err","").replace(".dat",""))
print(f"[!] CLEAN: successfull emulations: {len(X_clean_raw)}, errored: {len(X_clean_raw_error)}, success ratio: {len(X_clean_raw)*100/(len(X_clean_raw_error)+len(X_clean_raw)):.2f}%")

X = X_malicious_raw + X_clean_raw
y = np.hstack([np.ones(len(X_malicious_raw)),np.zeros(len(X_clean_raw))])

# TRAIN / VAL SPLIT
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=RANDOM_SEED)

# with open("X_train.pickle.set", "wb") as f:
#     pickle.dump(X_train, f)

# with open("X_val.pickle.set", "wb") as f:
#     pickle.dump(X_test, f)

# np.save("y_train", y_train)
# np.save("y_val", y_test)

# TEST OBJECTS
X, y = shuffle(X, y, random_state=RANDOM_SEED)

with open("X_test.pickle.set", "wb") as f:
    pickle.dump(X, f)

np.save("y_test", y)