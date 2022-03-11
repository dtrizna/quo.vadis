import sys
import time
import argparse
import logging

import pickle
import numpy as np

import torch

from sklearn.linear_model import LogisticRegression

sys.path.append("..") # repository root
sys.path.append(".")
from utils.hashpath import get_filepath_db, get_rawpe_db
from models import Composite

from models import Emulation, Filepath
from modules.sota.models import MalConvModel, EmberModel_2019

MALCONV_MODEL_PATH = '../modules/sota/malconv/parameters/malconv.checkpoint'
EMBER_2019_MODEL_PATH = '../modules/sota/ember/parameters/ember_model.txt'

FILEPATH_MODEL_PATH = '../modules/filepath/pretrained/1646930331-model.torch'
FILEPATH_BYTES = '../modules/filepath/pretrained/keep_bytes-ed64-pl150-kb150-1646917941.pickle'

EMULATION_MODEL_PATH = '../modules/emulation/pretrained/1646990611-model.torch'
EMULATION_APICALLS = '../modules/emulation/pretrained/api_calls_preserved-ed96-pl150-kb600-1646926097.pickle'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PADDING_LENGTH = 150

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    parser.add_argument("--how", type=str, nargs="+", default=["malconv", "paths", "emulation", "ember"], help="Specify space separated modules to use, e.g.: --how ember paths emulation")
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
        composite = Composite(modules=args.how, late_fusion_model="MLP")
    else:
        print("wrong model")
        sys.exit(1)

    
    # === PARSING PE TO VECTORS ===
    rawpe_db = get_rawpe_db()
    filepath_db = get_filepath_db()
    
    if args.train and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_train.pickle"
        args.y = "/data/quo.vadis/data/train_test_sets/y_train.npy"
    elif args.test and not args.pe_hashlist:
        args.pe_hashlist = "/data/quo.vadis/data/train_test_sets/X_test.pickle"
        args.y = "/data/quo.vadis/data/train_test_sets/y_test.npy"

    rawpe_db_hashlist = {}
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
        _ = [print(f"\t[Mean Sample Time] {k:>10}: {v:.4f}s") for k,v in composite.get_processing_time().items()]

    preds = composite.predict_proba(hashlist)
    print(preds, y)
        