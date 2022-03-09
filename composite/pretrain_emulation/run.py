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

sys.path.append("/data/quo.vadis/modules/morbus.certatio")
from model_train import rawseq2array, train
from preprocessing.reports import report_to_apiseq
from utils.functions import flatten
from models.classic import Modular


def dump_results(model, train_losses, train_metrics, duration):
    prefix = f"{int(time.time())}"
    model_file = f"{prefix}-model.torch"
    torch.save(model.state_dict(), model_file)
    
    with open(f"{prefix}-train_losses.pickle", "wb") as f:
        pickle.dump(train_losses, f)
    
    # in form [train_acc, train_f1]
    np.save(f"{prefix}-train_metrics.pickle", train_metrics)
    
    with open(f"{prefix}-duration.pickle", "wb") as f:
        pickle.dump(duration, f)

    dumpstring = f"""
     [!] {time.ctime()}: Dumped results:
            model: {model_file}
            train loss list: {prefix}-train_losses.pickle
            train metrics : {prefix}-train_metrics.pickle
            duration: {prefix}-duration.pickle"""
    logging.warning(dumpstring)


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

        if args.apis:
            logging.warning(f" [*] Loading preserved API calls from {args.apis}")
            with open(args.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
        else:
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
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.LongTensor(X),torch.LongTensor(y)),
        batch_size = 1024, shuffle=True)

    model = Modular(
        vocab_size = args.keep_apis + 2,
        embedding_dim = args.embedding_dim,
        # conv params
        filter_sizes = [2,3,4,5],
        num_filters = [128, 128, 128, 128],
        batch_norm_conv = False,

        # ffnn params
        hidden_neurons = [1024, 512, 256, 128],
        batch_norm_ffnn = True,
        dropout=0.5,
        num_classes=2,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    loss_function = nn.CrossEntropyLoss()

    # ========== MODEL TRAINING =============
    logging.warning(f" [*] Model training for {args.epochs} epochs...")
    
    train_losses = []
    train_metrics = []
    duration = []
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            logging.warning(f" [*] Started epoch: {epoch}")

            train_loss, train_m = train(model, device, train_loader, optimizer, loss_function, epoch, args.verbosity_batches)
            train_losses.extend(train_loss)
            train_metrics.append(train_m)

            # Print performance over the entire training data
            time_elapsed = time.time() - epoch_start_time
            duration.append(time_elapsed)
            logging.warning(f" [*] {time.ctime()}: {epoch + 1:^7} | Tr.loss: {np.mean(train_loss):^12.6f} | Tr.acc.: {np.mean([x[0] for x in train_m]):^9.2f} | Took: {time_elapsed:^9.2f}s")
        dump_results(model, train_losses, np.vstack(train_metrics), duration)
    except KeyboardInterrupt as ex:
        dump_results(model, train_losses, train_metrics, duration)
