import os
import time
import pickle
import logging
import shutil
import numpy as np
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.nn import functional as F

from preprocessing.array import rawseq2array, pad_array, byte_filter, remap
from preprocessing.text import normalize_path
from preprocessing.emulation import emulate
from preprocessing.reports import report_to_apiseq
from utils.hashpath import get_report_db, get_rawpe_db, get_filepath_db, get_filepath_from_hash
from utils.functions import sigmoid
from modules.sota.models import MalConvModel, EmberModel_2019

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


class Modular(nn.Module):
    def __init__(self, 
                # embedding params
                vocab_size = 152,
                embedding_dim = 32,
                # conv params
                filter_sizes = [2, 3, 4, 5],
                num_filters = [128, 128, 128, 128],
                batch_norm_conv = False,
                # ffnn params
                hidden_neurons = [128],
                batch_norm_ffnn = False,
                dropout = 0.5,
                num_classes = 2):
        super().__init__()
        
        # embdding
        self.embedding = nn.Embedding(vocab_size, 
                                  embedding_dim, 
                                  padding_idx=0)
        
        # convolutions
        self.conv1d_module = nn.ModuleList()
        for i in range(len(filter_sizes)):
                if batch_norm_conv:
                    module = nn.Sequential(
                                nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i]),
                                nn.BatchNorm1d(num_filters[i])
                            )
                else:
                    module = nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i])
                self.conv1d_module.append(module)

        # Fully-connected layers
        conv_out = np.sum(num_filters)
        self.ffnn = []

        for i,h in enumerate(hidden_neurons):
            self.ffnn_block = []
            if i == 0:
                self.ffnn_block.append(nn.Linear(conv_out, h))
            else:
                self.ffnn_block.append(nn.Linear(hidden_neurons[i-1], h))
            
            # add BatchNorm to every layer except last
            if batch_norm_ffnn:# and not i+1 == len(hidden_neurons):
                self.ffnn_block.append(nn.BatchNorm1d(h))
            
            self.ffnn_block.append(nn.ReLU())

            if dropout:
                self.ffnn_block.append(nn.Dropout(dropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnn_block))
        
        self.ffnn = nn.Sequential(*self.ffnn)
        self.fc_output = nn.Linear(hidden_neurons[-1], num_classes)
        self.relu = nn.ReLU()

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])
    
    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.conv_and_max_pool(embedded, conv1d) for conv1d in self.conv1d_module]
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fc_output(x_fc)
        
        return out


class Module(object):
    def __init__(self, device,
                    embedding_dim,
                    vocab_size,
                    state_dict=None, 
                    padding_length=150):
        self.device = device
        self.padding_length = padding_length
        self.model = Modular(
            vocab_size = vocab_size,
            embedding_dim = embedding_dim,
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


class Filepath(Module):
    def __init__(self, bytes, device, embedding_dim=64, state_dict=None):
        super().__init__(device, embedding_dim, len(bytes)+1, state_dict)
        self.bytes = bytes

    def evaluate_path(self, path):
        x = normalize_path(path).encode("utf-8", "ignore")
        x = np.array(list(x), dtype=int)
        x = pad_array(x, length=self.padding_length)
        x = byte_filter(x, [0,1] + self.bytes)

        orig_bytes = set([0,1]+sorted(self.bytes))
        mapping = dict(zip(orig_bytes, range(len(orig_bytes))))
        x = torch.LongTensor(remap(x, mapping)).reshape(1,-1)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        prediction = torch.argmax(logits, dim=1).flatten()
        
        return logits, prediction


class Emulation(Module):
    def __init__(self, apimap, device, embedding_dim=96, state_dict=None):
        super().__init__(device, embedding_dim, len(apimap)+2, state_dict)
        self.apimap = apimap
        self.report_db = get_report_db()

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
        
        success = emulate(path, temp_report_folder)
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

    def evaluate_report(self, h):
        report_fullpath = self.report_db[h]
        apiseq = report_to_apiseq(report_fullpath)["api.seq"]
        logits, preds = self.forwardpass_apiseq(apiseq)
        return logits, preds


class Composite(object):
    def __init__(self, modules=["malconv", "paths", "emulation", "ember"],
                    malconv_model_path = '../modules/sota/malconv/parameters/malconv.checkpoint',
                    ember_2019_model_path = '../modules/sota/ember/parameters/ember_model.txt',

                    filepath_model_path = '../modules/filepath/pretrained/1646930331-model.torch',
                    filepath_bytes = '../modules/filepath/pretrained/keep_bytes-ed64-pl150-kb150-1646917941.pickle',

                    emulation_model_path = '../modules/emulation/pretrained/1646990611-model.torch',
                    emulation_apicalls = '../modules/emulation/pretrained/api_calls_preserved-ed96-pl150-kb600-1646926097.pickle',
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    padding_length = 150,

                    late_fusion_model = "MLP"
                ):
        self.device = device
        self.padding_length = padding_length

        self.malconv_model_path = malconv_model_path
        self.ember_2019_model_path = ember_2019_model_path

        self.filepath_model_path = filepath_model_path
        self.emulation_model_path = emulation_model_path

        self.apis = emulation_apicalls
        self.bytes = filepath_bytes

        self.modules = {}

        if "malconv" in modules:
            self.modules["malconv"] = MalConvModel(self.malconv_model_path)
            
        if "ember" in modules:
            self.modules["ember"] = EmberModel_2019(self.ember_2019_model_path)
        
        if "paths" in modules:
            with open(self.bytes, "rb") as f:
                bytes = pickle.load(f)
            self.modules["paths"] = Filepath(bytes, self.device, state_dict=self.filepath_model_path)

        if "emulation" in modules:
            with open(self.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            self.apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
            self.modules["emulation"] = Emulation(self.apimap, self.device, state_dict=self.emulation_model_path)
        
        self.module_timers = [] # np.vstack(self.module_timers).mean(axis=0))

        self.rawpe_db = get_rawpe_db()
        self.filepath_db = get_filepath_db()

        if late_fusion_model == "LogisticRegression":
            self.model = LogisticRegression()
        elif late_fusion_model == "XGBClassifier":
            self.model = XGBClassifier(n_estimators=100, 
                                        objective='binary:logistic',
                                        eval_metric="logloss",
                                        use_label_encoder=False)
        elif late_fusion_model == "MLP":
            self.model = MLPClassifier() # just with default for now
        else:
            raise NotImplementedError


    def get_processing_time(self):
        return dict(zip(self.modules.keys(), np.vstack(self.module_timers).mean(axis=0)))

    def early_fusion_pass(self, h):
        vector = []
        checkpoint = []
        for model in self.modules:
            if model == "emulation":
                checkpoint.append(time.time())

                emul_logits, _ = self.modules["emulation"].evaluate_report(h)
                emul_prob = sigmoid(emul_logits.detach().numpy())[0,1]
                vector.append(emul_prob)
            
            if model == "paths":
                checkpoint.append(time.time())
            
                filepath = get_filepath_from_hash(h, self.filepath_db)
                path_logits, _ = self.modules["paths"].evaluate_path(filepath)
                path_prob = sigmoid(path_logits.detach().numpy())[0,1]
                vector.append(path_prob)
            
            if model == "ember":
                checkpoint.append(time.time())
            
                ember_prob = self.modules["ember"].get_score(self.rawpe_db[h])
                vector.append(ember_prob)
            
            if model == "malconv":
                checkpoint.append(time.time())
            
                malconv_prob = self.modules["malconv"].get_score(self.rawpe_db[h])
                vector.append(malconv_prob)
        
        checkpoint.append(time.time())
        self.module_timers.append([checkpoint[i]-checkpoint[i-1] for i in range(1,len(checkpoint))])
        
        return np.array(vector).reshape(1,-1)
    
    def fit_hashlist(self, hashlist, labels, dump_xy=False):
        x = []
        
        for i,h in enumerate(hashlist):
            print(f" [*] Scoring: {i+1}/{len(hashlist)}", end="\r")
            x.append(self.early_fusion_pass(h))
        
        self.x = np.vstack(x)
        self.y = labels

        if dump_xy:
            timestamp = int(time.time())
            np.save(f"./X-{timestamp}.npy", self.x)
            np.save(f"./y-{timestamp}.npy", self.y)
        
        self.model.fit(self.x, self.y)
    
    def predict_proba(self, hashlist):
        probs = []
        for h in hashlist:
            x = self.early_fusion_pass(h)
            probs.append(self.model.predict_proba(x))
        return np.vstack(probs)

    