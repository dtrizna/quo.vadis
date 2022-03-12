import sys
import argparse
import logging

import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

sys.path.append("..") # repository root
sys.path.append(".")
from models import Composite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    parser.add_argument("--how", type=str, nargs="+", default=["malconv", "filepaths", "emulation", "ember"], help="Specify space separated modules to use, e.g.: --how ember paths emulation")
    parser.add_argument("--model", type=str, default="mlp")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--pe", type=str, help="Path to the PE for evaluation")
    group1.add_argument("--pe-hashlist", type=str, help="Path to the PE for evaluation")
    parser.add_argument("--y", type=str, help="Path to the ground truth labels needed from training / evaluation")
    parser.add_argument("--save-xy", action="store_true")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--train", action="store_true", help="Whether to train fusion network")
    group2.add_argument("--test", action="store_true", help="Whether to evaluate fusion network")

    parser.add_argument("--limit", default=None, type=int, help="whether to limit parsing to some index (for testing)")

    args = parser.parse_args()


    # === LOADING MODEL ===
    if args.model == "lr":
        composite = Composite(modules=args.how, late_fusion_model="LogisticRegression")
    elif args.model == "xgb":
        composite = Composite(modules=args.how, late_fusion_model="XGBClassifier")
    elif args.model == "mlp":
        composite = Composite(modules=args.how, late_fusion_model="MultiLayerPerceptron")
    else:
        print("wrong late fusion model")
        sys.exit(1)

    
    # === PARSING PE TO VECTORS ===
    if args.train and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_train.pickle"
        args.y = "/data/quo.vadis/data/train_test_sets/y_train.npy"
    elif args.test and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_test.pickle"
        args.y = "/data/quo.vadis/data/train_test_sets/y_test.npy"

    if args.pe:
        h = args.pe.split("/")[-1]
        hashlist = [args.pe]

    elif args.pe_hashlist:
        with open(args.pe_hashlist, "rb") as f:
            hashlist = pickle.load(f)

    if args.limit:
        hashlist = hashlist[0:args.limit]
    y = np.load(args.y)[0:len(hashlist)]

    if not hashlist:
        print("Didn't load any data...")
        sys.exit()

    if args.train:
        composite.fit_hashlist(hashlist, y, dump_xy=args.save_xy)
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in composite.get_processing_time().items()]
        print("done")

    if args.test:
        composite.fit_hashlist(hashlist, y, dump_xy=args.save_xy)
        print()
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in composite.get_processing_time().items()]
        print("done")
    