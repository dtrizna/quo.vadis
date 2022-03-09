import sys
import os
import shutil
import logging
import argparse
import pickle

from pathlib import Path

import torch
from torch import nn, optim

sys.path.append("/data/quo.vadis/modules/morbus.certatio")
from models.classic import Modular
from model_train import rawseq2array, evaluate # evaluate(model, device, val_loader, loss_function)
from emulation.emulate_samples import emulate
from preprocessing.reports import report_to_apiseq
sys.path.remove("/data/quo.vadis/modules/morbus.certatio")

sys.path.append("/data/quo.vadis/modules/quo.vadis.primus")
# TBD
sys.path.remove("/data/quo.vadis/modules/quo.vadis.primus")

PADDING_LENGTH = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evalute_apiseq(apiseq, apimap, model):
    x = rawseq2array(apiseq, apimap, PADDING_LENGTH)
    x = torch.LongTensor(x.reshape(1,-1)).to(DEVICE)
    
    model.eval()
    logits = model(x)
    prediction = torch.argmax(logits, dim=1).flatten()

    return logits, prediction


def evaluate_rawpe(path, apimap, model, i=0, l=0):
    temp_report_folder = f"temp_reports"
    os.makedirs(temp_report_folder, exist_ok=True)
    
    success = emulate(path, i, l, temp_report_folder)
    if not success:
        logging.error(f" [-] Failed emulation of {path}")
    else:
        samplename = path.split("/")[-1]
        reportfile = f"{temp_report_folder}/{samplename}.json"
        apiseq = report_to_apiseq(reportfile)["api.seq"]
        logits, prediction = evalute_apiseq(apiseq, apimap, model)

    # cleanup
    shutil.rmtree(temp_report_folder)
    return logits, prediction


def evaluate_hash(h, apimap, model):
    try:
        report_fullpath = str([x for x in Path("/data/quo.vadis/data/emulation.dataset").rglob(h+".json")][0])
    except KeyError as ex:
        logging.error(f" [-] Cannot find report for: {h}... {ex}")

    apiseq = report_to_apiseq(report_fullpath)["api.seq"]
    logits, preds = evalute_apiseq(apiseq, apimap, model)

    return logits, preds
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    parser.add_argument("--bytes", type=str, help="Pickle serialized file with list of preserved bytes used for training")
    parser.add_argument("--path-model", type=str, help="Path to .toch file with pretrained FilePath model weights")
    parser.add_argument("--apis", type=str, help="Pickle serialized file with list of preserved API calls used for training")
    parser.add_argument("--emulation-model", type=str, help="Path to .toch file with pretrained Emulation model weights")

    args = parser.parse_args()
    
    if args.path_model:
        if args.bytes:
            logging.warning(f" [*] Loading frequent bytes from {args.bytes}")
            with open(args.bytes, "rb") as f:
                keep_bytes = pickle.load(f)
        else:
            logging.warning(f" [*] You need to provide preserved bytes pickle object to evaluate path")
            sys.exit(1)

        path_model = Modular(
            vocab_size = len(keep_bytes) + 1, # 'pad' already included
            embedding_dim = 96,
            # conv params
            filter_sizes = [2,3,4,5],
            num_filters = [128,128,128,128],
            batch_norm_conv = False,

            # ffnn params
            hidden_neurons = [1024,512,256,128],
            batch_norm_ffnn = True,
            dropout=0.5,
            num_classes=2,
            )
        path_model.to(DEVICE)
        logging.warning(f" [*] Loading PyTorch model state from {args.path_model} as FilePath model")
        path_model.load_state_dict(torch.load(args.path_model))

        # TBD path evaluate functions
    
    if args.emulation_model:
        if args.apis:
            logging.warning(f" [*] Loading preserved API calls from {args.apis}")
            with open(args.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
        else:
            logging.warning(f" [-] You need to provide preserved API pickle file!")
            sys.exit(1)

        example_hash = "0a0ab5a01bfed5818d5667b68c87f7308b9beeb2d74610dccf738a932419affd"
        example_pe = "/data/quo.vadis/data/pe.dataset/PeX86Exe/backdoor/0a0ab5a01bfed5818d5667b68c87f7308b9beeb2d74610dccf738a932419affd"

        emu_model = Modular(
            vocab_size = len(apimap) + 2,
            embedding_dim = 96,
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
        emu_model.to(DEVICE)
        logging.warning(f" [*] Loading PyTorch model state from {args.emulation_model} as Emulation model")
        emu_model.load_state_dict(torch.load(args.emulation_model))
    
        a,c = evaluate_hash(example_hash, apimap, emu_model)
        print(example_hash, a, c)
        b,d = evaluate_rawpe(example_pe, apimap, emu_model)
        print(example_pe, b, d)
    

