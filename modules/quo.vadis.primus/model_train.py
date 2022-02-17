import random
import time
import argparse
import logging
import pickle 
import sys
import os

import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score #, recall_score, precision_score

from preprocessing.text import load_txt
from preprocessing.array import byte_filter, remap

from models.classic import Modular


def set_seed(seed_value=1763):
    """Set seed for reproducibility."""
    logging.debug(f" [*] {time.ctime()}: Using random seed for all libraries: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, device, train_loader, optimizer, loss_function, epoch_id, verbosity_batches=100):
    model.train()

    train_metrics = []
    train_loss = []
    now = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        
        loss = loss_function(logits, target)
        train_loss.append(loss.item())
        
        loss.backward() # derivatives
        optimizer.step() # parameter update

        preds = torch.argmax(logits, dim=1).flatten()
        
        accuracy = (preds == target).cpu().numpy().mean() * 100
        f1 = f1_score(target, preds)
        #precision = precision_score(target, preds)
        #recall = recall_score(target, preds)
        train_metrics.append([accuracy, f1])#, precision, recall])
        
        if batch_idx % verbosity_batches == 0:
            logging.warning(" [*] {}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f} | Elapsed: {:.2f}s".format(
                time.ctime(), epoch_id, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), np.mean([x[0] for x in train_metrics]), time.time()-now))
            now = time.time()

    return train_loss, np.array(train_metrics).mean(axis=0).reshape(-1,2)


def evaluate(model, device, val_loader, loss_function):
    model.eval()

    val_metrics = []
    val_loss = []

    # For each batch in our validation set...
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            logits = model(data)
        
        loss = loss_function(logits, target)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == target).cpu().numpy().mean() * 100
        f1 = f1_score(target, preds)
        #precision = precision_score(target, preds)
        #recall = recall_score(target, preds)
        val_metrics.append([accuracy, f1])#, precision, recall])
        
    return val_loss, np.array(val_metrics).mean(axis=0).reshape(-1,2)


def dump_results(model, train_losses, train_metrics, val_losses, val_metrics, duration, args, epoch):
    prefix = f"ep{epoch}-optim_{args.optimizer}-lr{args.learning_rate}-l2reg{args.l2}-dr{args.dropout}"
    prefix += f"_arr-ed{args.embedding_dim}-kb{args.keep_bytes}-pl{args.padding_length}"
    prefix += f"_model-conv{args.num_filters}-bn_c{args.batch_norm_conv}_f{args.batch_norm_ffnn}-ffnn{'_'.join([str(x) for x in args.hidden_layers])}"
    
    model_file = f"{prefix}-model.torch"
    torch.save(model.state_dict(), model_file)

    with open(f"{prefix}-train_losses.pickle", "wb") as f:
        pickle.dump(train_losses, f)
    
    with open(f"{prefix}-val_losses.pickle", "wb") as f:
        pickle.dump(val_losses, f)
    
    # in form [train_acc, train_f1]
    np.save(f"{prefix}-train_metrics.pickle", train_metrics)

    # in form [val_acc, val_f1]
    np.save(f"{prefix}-val_metrics.pickle", val_metrics)

    with open(f"{prefix}-duration.pickle", "wb") as f:
        pickle.dump(duration, f)

    dumpstring = f"""
     [!] {time.ctime()}: Dumped results:
            model: {model_file}
            train loss list: {prefix}-train_losses.pickle
            validation loss list: {prefix}-val_losses.pickle
            train metrics : {prefix}-train_metrics.pickle
            validation metrics : {prefix}-train_metrics.pickle
            duration: {prefix}-duration.pickle"""
    logging.warning(dumpstring)


def read_txt_arguments(X, y, arg, prefix, benign):
    how = "benign" if benign else "malicious"
    
    if not prefix:
        filelist = arg
    else:
        filelist = []
        for prefix in arg:
            folder = "/".join(prefix.split("/")[:-1])
            prefix = prefix.split("/")[-1]
            files = [f"{folder}/{x}" for x in os.listdir(folder) if x.startswith(prefix)]
            filelist.extend(files)

    for txt in filelist:
        logging.debug(f" [*] {time.ctime()}: X shape: {X.shape}, y shape: {y.shape}")
        logging.warning(f" [*] {time.ctime()}: Loading data from a TXT file as {how}: {txt}")
        txtdata = load_txt(txt, padding_length=args.padding_length)
        X_txtdata = np.stack(txtdata.values)

        if benign:
            # label == 0
            y_txtdata = np.zeros(X_txtdata.shape[0])
        else:
            # label == 1
            y_txtdata = np.ones(X_txtdata.shape[0])

        X = np.vstack([X, X_txtdata])
        y = np.vstack([y.reshape(-1,1), y_txtdata.reshape(-1,1)]).squeeze()
    
    return X, y


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

    # training specifics
    parser.add_argument("--seed", type=int, default=1763, help="Random seed to use (for reproducibility purposes)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch Size to use during training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning Rate for optimization algorithm")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum if SGD or RMSProp optimization algorithms are used")
    parser.add_argument("-l2", type=float, default=0, help="Will enable L2-regularization, specify alpha hyperparameter") # if present, good values: 1e-2 or 1e-4
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate to use, 0 to skip Dropout whatsoever")
    
    # model specific
    parser.add_argument("--model-input", type=str, help="Provide path to existing (if have already trained model).")

    # embedding params
    parser.add_argument("--keep-bytes", type=int, default=150, help="Specify number of TOP N bytes to keep during preprocessing.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Size of embedding dimension.")
    parser.add_argument("--padding-length", type=int, default=150, help="Length of array representing a single flepath.")

    # conv params
    parser.add_argument("-fs","--filter-sizes", type=int, nargs="+", default=[2,3,4,5], help="Size of Convolutional filters, specify multiple for parallel Convolutional layers. Use as: -fs 1 2 3 4")
    parser.add_argument("--num-filters", type=int, default=128, help="Number of Convolutional filters per layer")
    parser.add_argument("--batch-norm-conv", action="store_true", help="Whether to apply Batch Normalization to convolutional layers")
    # ffnn params
    parser.add_argument("-ffnn","--hidden-layers", type=int, nargs="+", default=[128], help="Use as: -ffnn 256 128 64")
    parser.add_argument("--batch-norm-ffnn", action="store_true", help="Whether to apply Batch Normalization to ffnn layers")

    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--verbosity-batches", type=int, default=100, help="Output stats based after number of batches")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")
    parser.add_argument("--save-xy", action="store_true", help="Whether to dump X and y arrays to disk")

    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    # reproducibility
    set_seed(args.seed)

    # ========== DATA =========
    # Threat Intel Data in CSV
    start_preprocessing = time.time()

    if not args.x:
        X = np.empty((0, args.padding_length), dtype=int)
        y = np.empty((0,0), dtype=int)
    else:
        if args.y:
            logging.warning(f" [*] {time.ctime()}: Loading data from a NumPy array: {args.x}")
            X = np.load(args.x)
            y = np.load(args.y)
        else:
            logging.warning(f" [-] {time.ctime()}: Please provide label array by -y <file>")
            sys.exit(1)
        if args.padding_length != 150:
            logging.warning(f" [!] {time.ctime()}: Ignored provided argument of padding length: {args.padding_length}, since preprocessed arrays were submitted!")
            args.padding_length = X.shape[1]
    
    # TXT files
    if args.malicious_txt:
        X, y = read_txt_arguments(X, y, args.malicious_txt, prefix=False, benign=False)
    elif args.malicious_prefix:
        X, y = read_txt_arguments(X, y, args.malicious_prefix, prefix=True, benign=False)
    
    if args.benign_txt:
        X, y = read_txt_arguments(X, y, args.benign_txt, prefix=False, benign=True)
    elif args.benign_prefix:
        X, y = read_txt_arguments(X, y, args.benign_prefix, prefix=True, benign=True)

    logging.debug(f" [*] {time.ctime()}: X shape: {X.shape}, y shape: {y.shape}")

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
        # saving bytes to later use
        with open(f"keep_bytes_{int(time.time())}.pickle", "wb") as f:
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

    if args.save_xy:
        suffix = f"ed{args.embedding_dim}-pl{args.padding_length}-kb{args.keep_bytes}-{int(time.time())}"
        np.save(f"X-{suffix}", X)
        np.save(f"y-{suffix}", y)
        with open(f"keep_bytes-{suffix}.pickle", "wb") as f:
            pickle.dump(keep_bytes, f)

    logging.warning(f" [!] {time.ctime()}: Dataset: benign {y[y==0].shape[0]*100/y.shape[0]:.2f} %, malicious {y[y==1].shape[0]*100/y.shape[0]:.2f} % | Preprocessing took: {time.time()-start_preprocessing:.2f}s")

    # ========= CREATING TRAINING AND VALIDAITON LOADERS ==========
    # splitting for validation 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=args.seed)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.LongTensor(X_train),torch.LongTensor(y_train)),
        batch_size = args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.LongTensor(X_val),torch.LongTensor(y_val)),
        batch_size = args.batch_size, shuffle=True)

    # ========== MODEL DEFINITION =============

    model = Modular(
        vocab_size = len(keep_bytes) + 2,
        embedding_dim = args.embedding_dim,
        # conv params
        filter_sizes = args.filter_sizes,
        num_filters = [args.num_filters] * len(args.filter_sizes),
        batch_norm_conv = args.batch_norm_conv,

        # ffnn params
        hidden_neurons = args.hidden_layers,
        batch_norm_ffnn = args.batch_norm_ffnn,
        dropout=args.dropout,
        num_classes=2,
    )
    if args.model_input:
        logging.warning(f" [*] {time.ctime()}: Loading PyTorch model state from {args.model_input}")
        model.load_state_dict(torch.load(args.model_input))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.optimizer == "sgd": # default lr = 0.01 (or even 0.1), momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.l2)
    elif args.optimizer == "rmsprop": # default lr = 0.01, momentum = 0
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.l2)
    elif args.optimizer == "adamw": # default lr = 0.001
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    else: # default lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    loss_function = nn.CrossEntropyLoss()

    # ========== MODEL TRAINING =============
    logging.warning(f" [*] {time.ctime()}: Model training for {args.epochs} epochs...")
    
    train_losses = []
    train_metrics = []
    val_losses = []
    val_metrics = []
    duration = []
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            logging.warning(f" [*] {time.ctime()}: Started epoch: {epoch}")

            train_loss, train_m = train(model, device, train_loader, optimizer, loss_function, epoch, args.verbosity_batches)
            train_losses.extend(train_loss)
            train_metrics.append(train_m)

            # After the completion of each training epoch, measure the model's performance on validation set.
            val_loss, val_m = evaluate(model, device, val_loader, loss_function)
            val_losses.extend(val_loss)
            val_metrics.append(val_m)

            # Print performance over the entire training data
            time_elapsed = time.time() - epoch_start_time
            duration.append(time_elapsed)
            logging.warning(f" [*] {time.ctime()}: {epoch + 1:^7} | Tr.loss: {np.mean(train_loss):^12.6f} | Tr.acc.: {np.mean([x[0] for x in train_m]):^9.2f} | Val.loss: {np.mean(val_loss):^10.6f} | Val.acc.: {np.mean([x[0] for x in val_m]):^9.2f} | {time_elapsed:^9.2f}")
        dump_results(model, train_losses, np.vstack(train_metrics), val_losses, np.vstack(val_metrics), duration, args, epoch)
    except KeyboardInterrupt as ex:
        dump_results(model, train_losses, train_metrics, val_losses, val_metrics, duration, args, epoch)
