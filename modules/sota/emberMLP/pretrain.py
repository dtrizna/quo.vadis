import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import random
import pickle
import time
import os
import ember

from nets import EmberMLP, PENetwork, BasicMLP, PENetworkOrig

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import argparse
import logging


def load_my_ember(ember_feature_folder="./vectorize_output_1657954970", batch_size=1024):

    print("[*] Reading my vectors ... ")
    X_train = np.load(f"{ember_feature_folder}/X_ember_trainset.npy")
    y_train = np.load(f"{ember_feature_folder}/y_ember_trainset.npy")
    X_val = np.load(f"{ember_feature_folder}/X_ember_valset.npy")#.astype(np.int8)
    y_val = np.load(f"{ember_feature_folder}/y_ember_valset.npy")#.astype(np.int8)

    return X_train, y_train, X_val, y_val


def load_orig_ember(dump=False, read_filtered=True, batch_size=1024):
    
    print("[*] Reading original vectors ... ")
    if read_filtered:
        # ==== D - reading
        X_train = np.load(open("data-orig-ember-filtered/X_train_ember_0_and_1.npy", "rb"))
        y_train = np.load(open("data-orig-ember-filtered/y_train_ember_0_and_1.npy", "rb"))
        X_val = np.load(open("data-orig-ember-filtered/X_val_ember_0_and_1.npy", "rb"))
        y_val = np.load(open("data-orig-ember-filtered/y_val_ember_0_and_1.npy", "rb"))

    else:
        X_train, y_train, X_val, y_val = ember.read_vectorized_features("/data/quo.vadis/data/ember/ember2018/")
        print("[!] Filtering out unlabeled data..")

        # ===== A: Filtering out unlabeled samples
        # label, which may be 0 for benign, 1 for malicious, or -1 for unlabeled
        indices_train = np.where(y_train != -1)
        X_train = X_train[indices_train]
        y_train = y_train[indices_train]

        indices_val = np.where(y_val != -1)
        X_val = X_val[indices_val]
        y_val = y_val[indices_val]

    # ==== B: Dumping
    if dump:
        np.save(open("data-orig-ember-filtered/X_train_ember_0_and_1.npy", "wb"), X_train)
        np.save(open("data-orig-ember-filtered/y_train_ember_0_and_1.npy", "wb"), y_train)
        np.save(open("data-orig-ember-filtered/X_val_ember_0_and_1.npy", "wb"), X_val)
        np.save(open("data-orig-ember-filtered/y_val_ember_0_and_1.npy", "wb"), y_val)

    # ==== C: Deepcopy - failed -- for now arrays are read-only?
    # print("[!] Deepcopy of vectors ... X_train")
    # X_train = X_trainr.copy() # deepcopy(X_trainr)
    # print("[!] Deepcopy of vectors ... X_test")
    # X_val = X_valr.copy() # deepcopy(X_valr)
    # print("[!] Deepcopy of vectors ... y")
    # y_train, y_val = deepcopy(y_trainr), deepcopy(y_valr)

    return X_train, y_train, X_val, y_val


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return y_pred_tag.detach().numpy(), acc

    
def set_seed(seed_value=1763):
    """Set seed for reproducibility."""
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

        loss.backward() # derivatives
        optimizer.step() # parameter update

        train_loss.append(loss.item())
        
        pred_labels, accuracy = binary_acc(logits, target)
        f1 = f1_score(target, pred_labels)
        rocauc = roc_auc_score(target, pred_labels)
        train_metrics.append([accuracy, f1, rocauc])
        
        if batch_idx % verbosity_batches == 0:
            logging.warning(" [*] {}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f} | Elapsed: {:.2f}s".format(
                time.ctime(), epoch_id, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), np.mean([x[0] for x in train_metrics]), time.time()-now))
            now = time.time()

    return train_loss, np.array(train_metrics).mean(axis=0).reshape(-1,3), logits, target


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

        pred_labels, accuracy = binary_acc(logits, target)
        f1 = f1_score(target, pred_labels)
        rocauc = roc_auc_score(target, pred_labels)
        val_metrics.append([accuracy, f1, rocauc])
        
    return val_loss, np.array(val_metrics).mean(axis=0).reshape(-1,3)


def dump_results(model, train_losses, train_metrics, val_losses, val_metrics, duration, args, epoch):
    outdir = f"train_output_{int(time.time())}"
    os.mkdir(outdir)
    prefix = f"{outdir}/ep{epoch}-optim_{args.optimizer}-lr{args.learning_rate}-l2reg{args.l2}-dr{args.dropout}"
    
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    # data specifics
    parser.add_argument("--ember-feature-folder", type=str)
        
    # training specifics
    parser.add_argument("--seed", type=int, default=1763, help="Random seed to use (for reproducibility purposes)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch Size to use during training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning Rate for optimization algorithm")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum if SGD or RMSProp optimization algorithms are used")
    parser.add_argument("-l2", type=float, default=0, help="Will enable L2-regularization, specify alpha hyperparameter") # if present, good values: 1e-2 or 1e-4
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate to use, 0 to skip Dropout whatsoever")
    
    # model specific
    parser.add_argument("--model-input", type=str, help="Provide path to existing (if have already trained model).")

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
    EMBER_FEATURE_DIM = 2381

    # =================
    # TAKING ORIGINAL EMBER DATASET
    #X_train, y_train, X_val, y_val = load_orig_ember(read_filtered=True, dump=False, batch_size=1024)

    # =================
    # OUR DATASET
    #X_train_my, y_train_my, X_val_my, y_val_my = load_my_ember(ember_feature_folder=args.ember_feature_folder, batch_size=1024)

    # == ! ==
    # MERGED DATASET
    #print("[*] Merging datasets...")
    #X_train = np.vstack([X_train, X_train_my])
    #np.save(open("X_train_total.npy", "wb"), X_train)
    # X_val = np.vstack([X_val, X_val_my])
    # np.save(open("X_val_total.npy", "wb"), X_val)
    # del X_train_my, X_val_my
    
    # y_train = np.vstack([y_train.reshape(-1,1), y_train_my.reshape(-1,1)])
    # np.save(open("y_train_total.npy", "wb"), y_train)
    # y_val = np.vstack([y_val.reshape(-1,1), y_val_my.reshape(-1,1)])
    # np.save(open("y_val_total.npy", "wb"), y_val)
    # del y_train_my, y_val_my
    
    print("[*] Reading dataset...", end= " ")
    now = time.time()
    X_train = np.load(open("data-total/X_train_total.npy", "rb"))
    X_val = np.load(open("data-totalX_val_total.npy", "rb"))
    y_train = np.load(open("data-totaly_train_total.npy", "rb"))
    y_val = np.load(open("data-totaly_val_total.npy", "rb"))
    print(f"Took: {time.time() - now:.2f}s") # takes ~200 s

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size = args.batch_size, shuffle=True)

    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size = args.batch_size, shuffle=True)

    # MODELLING

    # this network ends with softmax
    # model = PENetworkOrig(feature_dimension=EMBER_FEATURE_DIM)
    # loss_function = nn.BCELoss()

    # these networks doesn't end with softmax
    model = PENetwork(feature_dimension=EMBER_FEATURE_DIM)
    # model = BasicMLP(input_dim=87)
    # model = EmberMLP(input_dim=EMBER_FEATURE_DIM)
    loss_function = nn.BCEWithLogitsLoss() 
    

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
    
    train_losses = []
    train_metrics = []
    val_losses = []
    val_metrics = []
    duration = []

    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            logging.warning(f" [*] {time.ctime()}: Started epoch: {epoch}")

            train_loss, train_m, logits, target = train(model, device, train_loader, optimizer, loss_function, epoch, args.verbosity_batches)
            train_losses.extend(train_loss)
            train_metrics.append(train_m)

            # After the completion of each training epoch, measure the model's performance on validation set.
            val_loss, val_m = evaluate(model, device, val_loader, loss_function)
            val_losses.extend(val_loss)
            val_metrics.append(val_m)

            # Print performance over the entire training data
            time_elapsed = time.time() - epoch_start_time
            duration.append(time_elapsed)
            logging.warning(f" [*] {time.ctime()}: {epoch:^7} | Tr.loss: {np.mean(train_loss):^12.6f} | Tr.acc.: {np.mean([x[0] for x in train_m]):^6.2f} | Val.loss: {np.mean(val_loss):^10.6f} | Val.acc.: {np.mean([x[0] for x in val_m]):^6.2f} | Took: {time_elapsed:^9.2f}s")
        dump_results(model, train_losses, np.vstack(train_metrics), val_losses, np.vstack(val_metrics), duration, args, epoch)
    except KeyboardInterrupt as ex:
        dump_results(model, train_losses, train_metrics, val_losses, val_metrics, duration, args, epoch)

