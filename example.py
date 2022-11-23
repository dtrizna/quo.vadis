import sys
import argparse

import pickle
import numpy as np

import os
import requests
import py7zr
import logging

repo_root = "./"
sys.path.append(repo_root)
from models import CompositeClassifier

def download_vxpe(link):
    localarchive = link.split("/")[-1]
    hhash = localarchive.replace(".7z", "")
    archive = requests.get(vx_link).content
    with open(localarchive, "wb") as f:
        f.write(archive)
    with py7zr.SevenZipFile(localarchive, "r", password='infected') as archive:
        bytez = archive.read(targets=hhash)[hhash].read()
    with open(hhash, "wb") as f:
        f.write(bytez)
    os.remove(localarchive)
    return "./"+hhash

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    parser.add_argument("--example", action="store_true", help="Just show an EXAMPLE -- download BoratRAT from vx-underground, analyze it, and exit\n")

    parser.add_argument("--how", type=str, nargs="+", default=["filepaths", "emulation", "ember"], help="Specify space separated modules to use, e.g.: --how ember paths emulation")
    parser.add_argument("--model", type=str, default="MultiLayerPerceptron", help="Options: LogisticRegression, XGBClassifier, MultiLayerPerceptron")


    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--pe-sample", type=str, help="Path to the PE sample")
    group1.add_argument("--pe-hashlist", type=str, help="Path to the PE pickle hashlist object")
    parser.add_argument("--y", type=str, help="Path to the ground truth labels needed from training / evaluation")
    parser.add_argument("--save-xy", action="store_true")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--train", action="store_true", help="Whether to train fusion network")
    group2.add_argument("--val", action="store_true", help="Whether to evaluate fusion network")
    group2.add_argument("--test", action="store_true", help="Whether to evaluate fusion network")

    parser.add_argument("--limit", default=None, type=int, help="whether to limit parsing to some index (for testing)")

    args = parser.parse_args()

    hashlist = None
    # === PARSING PE TO VECTORS ===
    # DBs
    filepath_csvs = "data/pe.dataset/PeX86Exe"
    emulation_report_path = "data/emulation.dataset"
    rawpe_db_path = "data/pe.dataset/PeX86Exe"

    if args.train and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_val_test_sets/X_train.pickle.set"
        args.y = "/data/quo.vadis/data/train_val_test_sets/y_train.arr"
    elif args.val and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_val_test_sets/X_val.pickle.set"
        args.y = "/data/quo.vadis/data/train_val_test_sets/y_val.arr"
    elif args.test and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_val_test_sets/X_test.pickle.set"
        args.y = "/data/quo.vadis/data/train_val_test_sets/y_test.arr"
        
        fielpath_csvs = "data/pe.dataset/testset/"
        emulation_report_path = "data/emulation.dataset/testset_emulation"
        rawpe_db_path = "data/pe.dataset/testset"

    # === LOADING MODEL ===
    print("\n[*] Loading model...")
    classifier = CompositeClassifier(modules=args.how, root=repo_root, 
                                        meta_model=args.model,
                                        emulation_report_path = emulation_report_path,
                                        rawpe_db_path = rawpe_db_path,
                                        load_meta_model = True,
                                        fielpath_csvs=filepath_csvs)

    if args.example:
        # BENIGN SAMPLES PRESENT IN REPOSITORY
        base_dir = os.path.join(os.path.dirname(__file__), "evaluation", "adversarial", "samples_goodware")
        files = [r"C:\Windows\system32\calc.exe"]
        paths = [r"C:\users\myuser\AppData\Local\Temp\exploit.exe"]
        
        print("\n[*] Legitimate 'calc.exe' analysis...")
        x = classifier.preprocess_pelist(files, takepath=True, dump_xy=False)
        probs = classifier.predict_proba(x)
        scores = classifier.get_module_scores(x)
        print(f"[!] Given path {files[0]}, probability (malware): {probs[:,1][0]:.6f}")
        print("[!] Individual module scores:\n\n", scores[classifier.modules.keys()],"\n")

        x = classifier.preprocess_pelist(files, pathlist=paths, dump_xy=False)
        probs = classifier.predict_proba(x)
        scores = classifier.get_module_scores(x)
        print(f"[!] Given path {paths[0]}, probability (malware): {probs[:,1][0]:.6f}")
        print("[!] Individual module scores:\n\n", scores[classifier.modules.keys()],"\n")
        
        
        # MALICIOUS SAMPLE - EXAMPLE WITH SINGLE SAMPLE
        # BoratRAT from VX-UNDERGROUND
        print("[*] BoratRAT analysis...")
        vx_link = "https://samples.vx-underground.org/samples/Families/BoratRAT/Samples/b47c77d237243747a51dd02d836444ba067cf6cc4b8b3344e5cf791f5f41d20e.7z"
        example_pe = download_vxpe(vx_link)
        
        try:
            example_path = r"%USERPROFILE%\Downloads\BoratRat.exe" # from VirusTotal
            pred, scores = classifier.predict_proba_pelist([example_pe], pathlist=[example_path], return_module_scores=True, dump_xy=False)
            print(f"\n[!] Given path {example_path}, probability (malware): {pred[:,1][0]:.4f}")
            print("[!] Individual module scores:\n\n", scores, "\n")

            example_path = r"C:\windows\system32\calc.exe"
            pred, scores = classifier.predict_proba_pelist([example_pe], pathlist=[example_path], return_module_scores=True, dump_xy=False)
            print(f"\n[!] Given path {example_path}, probability (malware): {pred[:,1][0]:.4f}")
            print("[!] Individual module scores:\n\n", scores, "\n")
        except FileNotFoundError:
            logging.warning(f"[-] Cannot access downloaded sample: {example_pe}.\n\t\tIsn't AV blocking / removed it? Be sure to add exclusion folder or disable AV. Exiting ... ")
        
        #os.remove(example_pe)
        sys.exit(0)


    if args.pe_sample:
        h = args.pe_sample.split("/")[-1]
        hashlist = [args.pe_sample]
    elif args.pe_hashlist:
        with open(args.pe_hashlist, "rb") as f:
            hashlist = pickle.load(f)
    if args.y:
        y = np.load(args.y)
    else:
        y = None

    if not hashlist and (args.train or args.test or args.val):
        logging.error("[-] Please define a data to load using --pe-sample or --pe-hashlist")
        sys.exit(1)
    
    if args.limit:
        hashlist = hashlist[0:args.limit]
        if y:
            y = y[0:len(hashlist)]

    if args.train:
        classifier.fit_pelist(hashlist, y, dump_xy=args.save_xy)
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in classifier.get_module_processing_time().items()]
        print("done")

    if args.val or args.test:
        x = classifier.preprocess_pelist(hashlist, dump_xy=args.save_xy, defaultpath="C:\\users\\user\\Downloads\\file")
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in classifier.get_module_processing_time().items()]
        print("done")
        
        # getting actual prediction scores of specific examples with gold labels - for ERROR correction
        scores = classifier.get_module_scores(x, y)
        scores["path"] = [classifier.modules["filepaths"].filepath_db[x] for x in hashlist]
        scores["pefile"] = [classifier.rawpe_db[x] for x in hashlist]
        scores.to_csv("test_scores.csv", index=False)
        print(scores)
