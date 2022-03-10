import sys
import pickle
import argparse
import numpy as np
import logging
import time
from collections import Counter

import torch
from torch import nn, optim

sys.path.append("../../..")
from models import Filepath
from utils.hashpath import get_hashpath_db, get_path_from_hash
from preprocessing.array import pad_array, byte_filter, remap
from preprocessing.text import normalize_path

sys.path.append("..")
from model_train import read_txt_arguments


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    # data specifics
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--malicious-txt", type=str, nargs="+", help="Path to TXT file with malicious path data per line (multiple files can be provided).")
    group.add_argument("--malicious-prefix", type=str, nargs="+", help="Prefix of TXT files with malicious path data per line (multiple files can be provided).")
    
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--benign-txt", type=str, nargs="+", help="Path to TXT file with benign path data per line (multiple files can be provided).")
    group2.add_argument("--benign-prefix", type=str, nargs="+", help="Prefix of TXT files with benign path data per line (multiple files can be provided).")

    parser.add_argument("-x", type=str, help="Path to preprocessed input array")
    parser.add_argument("-y", type=str, help="Path to preprocessed input label array (provide only if X is given)")
    parser.add_argument("--bytes", type=str, help="Pickle serialized file with list of frequent bytes used for training")

    # embedding params
    parser.add_argument("--keep-bytes", type=int, default=150, help="Specify number of TOP N bytes to keep during preprocessing.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Size of embedding dimension.")
    parser.add_argument("--padding-length", type=int, default=150, help="Length of array representing a single flepath.")

    # auxiliary
    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")
    parser.add_argument("--verbosity-batches", type=int, default=100, help="Output stats based after number of batches")
    parser.add_argument("--save-xy", action="store_true", help="Whether to dump X and y arrays to disk")

    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train")

    args = parser.parse_args()
    
    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    # getting from files if provided
    PADDING_LENGTH = args.padding_length
    KEEP_BYTES = args.keep_bytes

    # ========= PREPROCESSING =========
    start_preprocessing = time.time()
    # X,y
    if not args.x:
        # getting from splitted data
        with open("/data/quo.vadis/data/train_test_sets/X_train.pickle", "rb") as f:
            X_train_hashes = pickle.load(f)
        l = len(X_train_hashes)
        hashpath_db = get_hashpath_db()
        fixed_vectors = []
        now = time.time()
        for i, h in enumerate(X_train_hashes):
            print(f" [*] Working on train PE set: {i}/{l}", end="\r")
            path = get_path_from_hash(h, hashpath_db)
            normalized_path_as_bytes = normalize_path(path).encode("utf-8", "ignore")
            vector = np.array(list(normalized_path_as_bytes), dtype=int)
            fixed_vectors.append(pad_array(vector, PADDING_LENGTH))
        
        X = np.vstack(fixed_vectors)
        y = np.load("/data/quo.vadis/data/train_test_sets/y_train.npy")
        logging.warning(f" [!] Train set loaded, took: {time.time() - now:.2f}s")

    else:
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

    # txt
    if args.malicious_txt:
        X, y = read_txt_arguments(X, y, args.malicious_txt, prefix=False, benign=False, padding_length=PADDING_LENGTH)
    elif args.malicious_prefix:
        X, y = read_txt_arguments(X, y, args.malicious_prefix, prefix=True, benign=False, padding_length=PADDING_LENGTH)
    
    if args.benign_txt:
        X, y = read_txt_arguments(X, y, args.benign_txt, prefix=False, benign=True, padding_length=PADDING_LENGTH)
    elif args.benign_prefix:
        X, y = read_txt_arguments(X, y, args.benign_prefix, prefix=True, benign=True, padding_length=PADDING_LENGTH)

    if args.bytes:
        logging.warning(f" [*] {time.ctime()}: Loading frequent bytes from {args.bytes}")
        with open(args.bytes, "rb") as f:
            keep_bytes = pickle.load(f)
        if args.keep_bytes != 150:
                    logging.warning(f" [!] {time.ctime()}: Ignored provided argument of keep bytes length: {args.keep_bytes}, since file to load bytes was provided!")
                    args.keep_bytes = len(keep_bytes)
    else:
        logging.warning(f" [*] {time.ctime()}: Initiating byte fequency analysis...")
        byte_counter = Counter([byte for sample in X for byte in sample]) # 4s
        # +1 since padding label was added within 'load_txt'
        n_bytes_to_keep = args.keep_bytes + 1
        keep_bytes = [x[0] for x in byte_counter.most_common(n_bytes_to_keep)] # 0.5s

    # Saving X before byte filter applied
    if args.save_xy:
        suffix = f"ed{args.embedding_dim}-pl{args.padding_length}-kb{args.keep_bytes}-{int(time.time())}"
        np.save(f"X-{suffix}", X)
        np.save(f"y-{suffix}", y)
        with open(f"keep_bytes-{suffix}.pickle", "wb") as f:
            pickle.dump(keep_bytes, f)

    # filtering only most common 'keep_bytes'
    # replaces rare bytes with [1]
    logging.warning(f" [*] {time.ctime()}: Initiating byte filter...")
    X = byte_filter(X, keep_bytes) # 3.5s

    # remapping for embedding: ~5 s
    try:
        logging.warning(f" [*] {time.ctime()}: Initiating byte remapping (needed for embeddings)...")
        orig_bytes = set([0,1]+sorted(keep_bytes[1:]))
        mapping = dict(zip(orig_bytes, range(len(orig_bytes))))
        # if you want map back, use:
        # remap(X, {v:k for k,v in mapping.items()})
        X = remap(X, mapping)
    except Exception as ex:
        print(ex)
        import pdb;pdb.set_trace()

    logging.warning(f" [!] {time.ctime()}: Dataset: benign {y[y==0].shape[0]*100/y.shape[0]:.2f} %, malicious {y[y==1].shape[0]*100/y.shape[0]:.2f} % | Preprocessing took: {time.time()-start_preprocessing:.2f}s")
    # ========= TRAINING =========
    logging.warning(f" [*] {time.ctime()}: Model training for {args.epochs} epochs...")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.LongTensor(X), torch.LongTensor(y)),
        batch_size = 1024, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quovadis = Filepath(keep_bytes, device, embedding_dim=args.embedding_dim)
    optimizer = optim.Adam(quovadis.model.parameters(), lr=1e-3, weight_decay=0)
    loss_function = nn.CrossEntropyLoss()
    quovadis.fit(args.epochs, optimizer, loss_function, train_loader)