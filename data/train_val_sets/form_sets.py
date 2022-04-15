from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

emulation_dataset_path = "/data/quo.vadis/data/emulation.dataset/"
emulation_dataset_folders_malicious = [x for x in os.listdir(emulation_dataset_path) if "report_" in x and "clean" not in x]
emulation_dataset_folders_clean = [x for x in os.listdir(emulation_dataset_path) if "report_" in x and "clean" in x]

X_malicious_raw = []
for folder in emulation_dataset_folders_malicious:
    for file in os.listdir(emulation_dataset_path + folder):
        if file.endswith(".json"):
            X_malicious_raw.append(file.rstrip(".json"))

X_clean_raw = []
for folder in emulation_dataset_folders_clean:
    for file in os.listdir(emulation_dataset_path + folder):
        if file.endswith(".json"):
            X_clean_raw.append(file.rstrip(".json"))

X = X_malicious_raw + X_clean_raw
y = np.hstack([np.ones(len(X_malicious_raw)),np.zeros(len(X_clean_raw))])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1763)

with open("X_train.pickle.set", "wb") as f:
    pickle.dump(X_train, f)

with open("X_val.pickle.set", "wb") as f:
    pickle.dump(X_test, f)

np.save("y_train", y_train)
np.save("y_val", y_test)
