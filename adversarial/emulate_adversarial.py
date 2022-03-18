
import os
import magic
import time

repo_root = "/data/quo.vadis/"
import sys
sys.path.append(repo_root)

from preprocessing.emulation import emulate
import threading

adversarial_testset_path = repo_root + "adversarial/samples_adversarial_testset_gamma_malconv/"
testset_files = os.listdir(adversarial_testset_path)

adversarial_fullpaths = []
# x_test_adv_paths = []
# y_test_adv = []
l = len(testset_files)
for i, file in enumerate(testset_files):
    print(f" [*] enumerating folder: {i}/{l}", end="\r")
    mm = magic.from_file(adversarial_testset_path + file)
    if "PE32" in mm:
        adversarial_fullpaths.append(adversarial_testset_path + file)
        #x_test_adv_paths.append(adversarial_testset_path + file)
        #y_test_adv.append(1)
    #else:
        #path = mm.replace("symbolic link to `", "").strip("'")
        #x_test_adv_paths.append(path)
        #if "clean" in path:
        #    y_test_adv.append(0)
        #else:
        #    y_test_adv.append(1)

#y_test_adv = np.array(y_test_adv, dtype=int)
#x_test_adv = classifier.preprocess_hashlist(x_test_adv_paths, dump_xy=True)

adversarial_emulation_folder = repo_root + "data/adversarial.emulation.dataset/reports_malconv/"
os.makedirs(adversarial_emulation_folder, exist_ok=True)

l = len(adversarial_fullpaths)
print(f" [!] Total size of adversarial samples to emulate: {l}")
for i, path in enumerate(adversarial_fullpaths):
    #emulate(path, adversarial_emulation_folder, i, l)
    thread = threading.Thread(target=emulate, args=(path, adversarial_emulation_folder, i, l))
    thread.start()
    while len(threading.enumerate()) > 20:
        time.sleep(0.1)