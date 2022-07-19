
import time
import argparse
import pickle
import logging

import numpy as np
from representation_model import CompositeClassifierFromRepresentations

def dump_xy(x, y=None, note=""):
    timestamp = int(time.time())
    xfile = f"./X{note}-{timestamp}.npy"
    np.save(xfile, x)
    logging.warning(f" [!] Dumped module scores to '{xfile}'")
    if y is not None:
        yfile = f"./y{note}-{timestamp}.npy"
        np.save(yfile, y)
        logging.warning(f" [!] Dumped module labels to '{yfile}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")
    parser.add_argument("--save-xy", action="store_true", help="Whether to dump X and y arrays to disk")

    parser.add_argument("--limit", type=int, help="Limit for tests")

    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    ROOT = "../../"

    train_valset_folder = ROOT + "data/train_val_test_sets/"

    X_train_hashes = pickle.load(open(f"{train_valset_folder}/X_train.pickle.set","rb"))
    y_train = np.load(open(f"{train_valset_folder}/y_train.arr","rb"))

    X_val_hashes = pickle.load(open(f"{train_valset_folder}/X_val.pickle.set","rb"))
    y_val = np.load(open(f"{train_valset_folder}/y_val.arr","rb"))

    model = CompositeClassifierFromRepresentations(root=ROOT)
    
    LIMIT = args.limit if args.limit else None
    
    logging.warning("[*] Preprocessing train set...")
    x_train = model.preprocess_pelist(X_train_hashes[0:LIMIT], dump_xy=False)

    if args.save_xy:
        dump_xy(x_train, y_train[:LIMIT], note="train")

    logging.warning("[*] Preprocessing validation set")
    x_val = model.preprocess_pelist(X_val_hashes[0:LIMIT], dump_xy=False)

    if args.save_xy:
        dump_xy(x_val, y_val[:LIMIT], note="val")
    
    if args.debug and args.limit:

        model.model.fit(x_train.detach().numpy(), y_train[0:LIMIT])
        
        probs_train = model.predict_proba(x_train.detach().numpy())
        preds_train = np.argmax(probs_train, axis=1)
        
        probs_val = model.predict_proba(x_val.detach().numpy())
        preds_val = np.argmax(probs_val, axis=1)
        
        print("\ntrain\n", preds_train, "\n", y_train[0:LIMIT].astype(int))
        print("\nval\n", preds_val, "\n", y_val[0:LIMIT].astype(int))

    import pdb;pdb.set_trace()