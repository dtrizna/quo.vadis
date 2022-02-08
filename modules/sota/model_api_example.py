import os
from sota import MalConvModel, EmberModel_2019


MALCONV_MODEL_PATH = 'malconv/parameters/malconv.checkpoint'
EMBER_2019_MODEL_PATH = 'ember/parameters/ember_model.txt'
MALWARE_PATH = '../data/pe.dataset/ransomware/'

for i,malware_example in enumerate(os.listdir(MALWARE_PATH)):
    malware_example = os.path.join(MALWARE_PATH, malware_example)

    print("[!] Example malware file: ", malware_example)

    m = MalConvModel(MALCONV_MODEL_PATH)
    print(f"[!] MalConv score: {m.get_score(malware_example)}")
    #print(f"\tEvasive? {m.is_evasive(malware_example)}")

    e = EmberModel_2019(EMBER_2019_MODEL_PATH)
    print(f"[!] Ember score: {e.get_score(malware_example)}")
    #print(f"\tEvasive? {e.is_evasive(malware_example)}")

    print("=="*50)