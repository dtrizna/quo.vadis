import sys
import argparse

import pickle
import numpy as np

repo_root = "../../"
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
    group2.add_argument("--test", action="store_true", help="Whether to evaluate fusion network")

    parser.add_argument("--limit", default=None, type=int, help="whether to limit parsing to some index (for testing)")

    args = parser.parse_args()


    # === LOADING MODEL ===
    composite = CompositeClassifier(modules=args.how, late_fusion_model=args.model, repo_root=repo_root)
    
    # === PARSING PE TO VECTORS ===
    if args.train and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_train.set"
        args.y = "/data/quo.vadis/data/train_test_sets/y_train.arr"
    elif args.test and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_test.set"
        args.y = "/data/quo.vadis/data/train_test_sets/y_test.arr"

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
        composite.fit_pelist(hashlist, y, dump_xy=args.save_xy)
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in composite.get_processing_time().items()]
        print("done")

    if args.test:
        composite.fit_pelist(hashlist, y, dump_xy=args.save_xy)
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in composite.get_processing_time().items()]
        print("done")
    