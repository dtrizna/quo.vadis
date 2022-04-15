import sys
import argparse

import pickle
import numpy as np

repo_root = "/data/quo.vadis/"
sys.path.append(repo_root)
from models import CompositeClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    parser.add_argument("--how", type=str, nargs="+", default=["malconv", "filepaths", "emulation", "ember"], help="Specify space separated modules to use, e.g.: --how ember paths emulation")
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
    classifier = CompositeClassifier(modules=args.how, root=repo_root, 
                                                        late_fusion_model=args.model, 
                                                        emulation_report_path = emulation_report_path,
                                                        rawpe_db_path = rawpe_db_path,
                                                        load_late_fusion_model = True,
                                                        fielpath_csvs=fielpath_csvs)

    if args.pe_sample:
        h = args.pe_sample.split("/")[-1]
        hashlist = [args.pe_sample]
    elif args.pe_hashlist:
        with open(args.pe_hashlist, "rb") as f:
            hashlist = pickle.load(f)
    
    if not hashlist:
        print("Didn't load any data...")
        sys.exit()
    
    if args.limit:
        hashlist = hashlist[0:args.limit]
    y = np.load(args.y)[0:len(hashlist)]

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
        
        # EXAMPLE API with single sample
        example_path = r"C:\windows\temp\kernel32.exe"
        example_pe = "/data/quo.vadis/data/pe.dataset/PeX86Exe/backdoor/0a0ab5a01bfed5818d5667b68c87f7308b9beeb2d74610dccf738a932419affd"

        pred, scores = classifier.predict_proba_pelist([example_pe], pathlist=[example_path], return_module_scores=True)
        print(f"Given path {example_path}, probability (malware): {pred[:,1][0]:.4f}")
        print(scores, "\n")

        example_path = r"C:\windows\system32\kernel32.dll"
        pred, scores = classifier.predict_proba_pelist([example_pe], pathlist=[example_path], return_module_scores=True)
        print(f"Given path {example_path}, probability (malware): {pred[:,1][0]:.4f}")
        print(scores)