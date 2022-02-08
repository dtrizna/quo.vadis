import time
import argparse
import logging
import pickle 
import torch
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")

from preprocessing.text import normalize_path
from preprocessing.array import byte_filter, fix_length, remap
from utils.functions import sigmoid
from models.classic import Net


def evaluate_sample(model, data):
    """Returns logits for a single filepath sample.

    Args:
        model (torch.nn.Module): PyTorch model that provides logits on ouput.
        data (np.array): Pre-processed (encoded, padded, etc.) filepath sample.

    Returns:
        torch.Tensor() : Tensor with outputs from model. 
    """
    model.eval()
    with torch.no_grad():
        # logit of prob: ln(p/(1-p))
        logits = model(data)
    return logits


def evaluate_list(model, input_list, keep_bytes, padding_length=150):
    """Function takes 

    Args:
        model (torch.nn.Module): PyTorch model that provides logits on ouput.
        input_list (list): List of filepath commands as str.
        keep_bytes (list): List of bytes that were used during training set encoding.

    Returns:
        list: List of np.arrays with probabilities [prob_benign, prob_malicious]
    """
    # define array and fill with contents from list
    X = np.zeros(shape=(len(input_list), padding_length), dtype=int)
    for i,x in enumerate(input_list):
        x = normalize_path(x).encode("utf-8", "ignore")
        x = np.array(list(x), dtype=int)
        X[i,:] = fix_length(x, length=padding_length)

    # keepign only frequent bytes & remapping 
    X = byte_filter(X, [0,1]+keep_bytes)
    orig_bytes = set([0,1]+sorted(keep_bytes))
    mapping = dict(zip(orig_bytes, range(len(orig_bytes))))
    X = remap(X, mapping)

    prob_list = []
    for i in range(X.shape[0]):
        x = torch.LongTensor(X[i,:]).reshape(1,-1)
        
        logits = evaluate_sample(model, x)
        probs = sigmoid(logits).numpy().squeeze()
        prob_list.append(probs)
    
    return prob_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    
    # model specific
    parser.add_argument("--model", type=str, help="Path to model state dictionary.")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("-fs","--filter-sizes", type=int, nargs="+", default=[2,3,4,5], help="Use as: -fs 1 2 3 4")
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    # preprocessing
    parser.add_argument("--bytes", required=True, type=str, help="Pickle serialized file with list of frequent bytes used for training.")
    parser.add_argument("--padding-length", type=int, default=150)

    # auxiliary
    parser.add_argument("--logfile", type=str, help="File to store logging messages.")
    parser.add_argument("--debug", help="Provide with DEBUG level information from packages.")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    # load frequent byte list
    logging.warning(f" [*] {time.ctime()}: Loading frequent bytes from {args.bytes}")
    with open(args.bytes, "rb") as f:
        keep_bytes = pickle.load(f)

    # TODO: get list of inputs from HTTP server
    input_list = ["C:\windows\sample.exe", "C:\windows\system32\cmd.exe", r"C:\users\test\desktop\ponny.pdf", r"C:\users\test\appdata\ponny.exe"]

    # instantiate pretrained model
    model = Net(
        vocab_size = len(keep_bytes) + 2,
        embedding_dim = args.embedding_dim,
        filter_sizes = args.filter_sizes,
        num_filters = [args.num_filters] * len(args.filter_sizes),
        dropout=args.dropout,
        num_classes=2
    )
    model.to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    prob_list = evaluate_list(model, input_list, keep_bytes, padding_length=args.padding_length)
    _ = [logging.warning(f"\t{input_list[i]: <50}: prob(malware) = {probs[1]:.4f}") for i,probs in enumerate(prob_list)]
