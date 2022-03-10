import sys
import os
import time
import pickle
import logging
import shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

import torch
from .classic import Modular

sys.path.append("..")
from preprocessing.array import rawseq2array
from emulation.emulate_samples import emulate
from preprocessing.reports import report_to_apiseq

class MorbusCertatio:
    def __init__(self, apimap, device, 
                    state_dict=None, 
                    padding_length=150):
        self.device = device
        self.apimap = apimap

        self.padding_length = padding_length
        
        self.model = Modular(
            vocab_size = len(self.apimap) + 2,
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
        self.model.to(device)
        
        if state_dict:
            self.model.load_state_dict(torch.load(state_dict))

    @staticmethod
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


    def forwardpass_apiseq(self, apiseq):
        x = rawseq2array(apiseq, self.apimap, self.padding_length)
        x = torch.LongTensor(x.reshape(1,-1)).to(self.device)

        self.model.eval()
        logits = self.model(x)
        prediction = torch.argmax(logits, dim=1).flatten()

        return logits, prediction

    def evaluate_rawpe(self, path, i=0, l=0):
        temp_report_folder = f"temp_reports"
        os.makedirs(temp_report_folder, exist_ok=True)
        
        success = emulate(path, i, l, temp_report_folder)
        if not success:
            logging.error(f" [-] Failed emulation of {path}")
        else:
            samplename = path.split("/")[-1]
            reportfile = f"{temp_report_folder}/{samplename}.json"
            apiseq = report_to_apiseq(reportfile)["api.seq"]
            logits, prediction = self.forwardpass_apiseq(apiseq)

        # cleanup
        shutil.rmtree(temp_report_folder)
        return logits, prediction


    def evaluate_report_by_hash(self, h):
        emulation_reports = "/data/quo.vadis/data/emulation.dataset"
        try:
            report_fullpath = str([x for x in Path(emulation_reports).rglob(h+".json")][0])
        except KeyError as ex:
            logging.error(f" [-] Cannot find report for: {h}... {ex}")

        apiseq = report_to_apiseq(report_fullpath)["api.seq"]
        logits, preds = self.forwardpass_apiseq(apiseq)

        return logits, preds


    @staticmethod
    def train(self, train_loader, optimizer, loss_function, epoch_id, verbosity_batches):
        self.model.train()

        train_metrics = []
        train_loss = []
        now = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            logits = self.model(data)

            loss = loss_function(logits, target)
            train_loss.append(loss.item())

            loss.backward() # derivatives
            optimizer.step() # parameter update  

            preds = torch.argmax(logits, dim=1).flatten()     
            
            accuracy = (preds == target).cpu().numpy().mean() * 100
            f1 = f1_score(target, preds)
            train_metrics.append([accuracy, f1])
            
            if batch_idx % verbosity_batches == 0:
                logging.warning(" [*] {}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f} | Elapsed: {:.2f}s".format(
                    time.ctime(), epoch_id, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), np.mean([x[0] for x in train_metrics]), time.time()-now))
                now = time.time()
        
        return train_loss, np.array(train_metrics).mean(axis=0).reshape(-1,2)


    def fit(self, epochs, optimizer, loss_function, train_loader, verbosity_batches=100):
        train_losses = []
        train_metrics = []
        duration = []

        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                logging.warning(f" [*] Started epoch: {epoch}")

                train_loss, train_m = self.train(self, train_loader, optimizer, loss_function, epoch, verbosity_batches)
                train_losses.extend(train_loss)
                train_metrics.append(train_m)

                time_elapsed = time.time() - epoch_start_time
                duration.append(time_elapsed)
                logging.warning(f" [*] {time.ctime()}: {epoch + 1:^7} | Tr.loss: {np.mean(train_loss):^12.6f} | Tr.acc.: {np.mean([x[0] for x in train_m]):^9.2f} | {time_elapsed:^9.2f}")
            self.dump_results(self.model, train_losses, np.vstack(train_metrics), duration)
        
        except KeyboardInterrupt:
            self.dump_results(self.model, train_losses, train_metrics, duration)


    def evaluate(self, val_loader, loss_function):
        self.model.eval()
    
        val_metrics = []
        val_loss = []
        # For each batch in our validation set...
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                logits = self.model(data)
            
            loss = loss_function(logits, target)
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == target).cpu().numpy().mean() * 100
            f1 = f1_score(target, preds)
            val_metrics.append([accuracy, f1])
            
        return val_loss, np.array(val_metrics).mean(axis=0).reshape(-1,2)
