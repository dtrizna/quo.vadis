import pickle
import logging
import time
import argparse
import sys
from pathlib import Path
import numpy as np
from collections import Counter

import torch
from torch import nn, optim

sys.path.append("../../../")
from models import Emulation
from utils.functions import flatten
from preprocessing.array import rawseq2array
from preprocessing.reports import report_to_apiseq

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    parser.add_argument("-x", type=str, help="Path to preprocessed input array")
    parser.add_argument("-y", type=str, help="Path to preprocessed input label array (provide only if X is given)")
    parser.add_argument("--apis", type=str, help="Pickle serialized file with list of preserved API calls used for training")

    # embedding params
    parser.add_argument("--keep-apis", type=int, default=600, help="Specify number of TOP N API calls to keep during preprocessing.")
    parser.add_argument("--embedding-dim", type=int, default=96, help="Size of embedding dimension.")
    parser.add_argument("--padding-length", type=int, default=150, help="Length of array representing a single emulation report.")

    # auxiliary
    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--verbosity-batches", type=int, default=100, help="Output stats based after number of batches")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")
    parser.add_argument("--save-xy", action="store_true", help="Whether to dump X and y arrays to disk")

    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")

    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    # ========= PREPROCESSING =========
    start_preprocessing = time.time()
    
    if args.apis:
        logging.warning(f" [*] Loading preserved API calls from {args.apis}")
        with open(args.apis, "rb") as f:
            api_calls_preserved = pickle.load(f)
        apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
    else:
        apimap = None

    if args.x:
        if args.y:
            logging.warning(f" [*] Loading data from a NumPy array: {args.x}")
            X = np.load(args.x)
            y = np.load(args.y)
        else:
            logging.warning(f" [-] Please provide label array by -y <file>")
            sys.exit(1)
        if args.padding_length != 150:
            logging.warning(f" [!] Ignored provided argument of padding length: {args.padding_length}, since preprocessed arrays were submitted!")
            args.padding_length = X.shape[1]
    else:
        # getting from splitted data
        with open("/data/quo.vadis/composite/dataset/X_train.pickle", "rb") as f:
            X_train_hashes = pickle.load(f)
        
        l = len(X_train_hashes)
        X_raw = []
        for i, h in enumerate(X_train_hashes):
            print(f" [*] Loading API call sequences: {i}/{l}", end="\r")
            
            try:
                report_fullpath = str([x for x in Path("/data/quo.vadis/data/emulation.dataset").rglob(h+".json")][0])
            except KeyError as ex:
                print(f" [-] Cannot find report for: {h}... Investigate!")
                sys.exit() # Cannot proceed since it screws y labels. If report not found - this is unacceptable.
            
            X_raw.append(report_to_apiseq(report_fullpath)["api.seq"])

        if not apimap:
            logging.warning(f" [*] Initiating API call analysis, preserving {args.keep_apis} most common calls")
            api_counter = Counter(flatten(X_raw))
            api_calls_preserved = [x[0] for x in api_counter.most_common(args.keep_apis)]
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))

        nowseq2arr = time.time()
        X = np.vstack([rawseq2array(x, apimap, args.padding_length) for x in X_raw])
        print(f" [!] Encoded raw API call sequences to vectors. Encoding took: {time.time() - nowseq2arr:.2f}s")
        y = np.load("/data/quo.vadis/composite/dataset/y_train.npy")
        
    print(X.shape, y.shape)

    logging.warning(f"[!] Preprocessing finished! Took: {time.time()-start_preprocessing:.2f}s")
    logging.warning(f"[!] Dataset: benign {y[y==0].shape[0]*100/y.shape[0]:.2f} %, malicious {y[y==1].shape[0]*100/y.shape[0]:.2f} %")

    if args.save_xy:
        suffix = f"ed{args.embedding_dim}-pl{args.padding_length}-kb{args.keep_apis}-{int(time.time())}"
        np.save(f"X-{suffix}", X)
        np.save(f"y-{suffix}", y)
        with open(f"api_calls_preserved-{suffix}.pickle", "wb") as f:
            pickle.dump(api_calls_preserved, f)
        
    # ========== TRAINING ===========
    logging.warning(f" [*] Model training for {args.epochs} epochs...")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.LongTensor(X),torch.LongTensor(y)),
        batch_size = 1024, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    morbus_certatio = Emulation(apimap, device)
    optimizer = optim.Adam(morbus_certatio.model.parameters(), lr=1e-3, weight_decay=0)
    loss_function = nn.CrossEntropyLoss()    
    morbus_certatio.fit(args.epochs, optimizer, loss_function, train_loader)