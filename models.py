import os
import sys
import time
import pickle
import logging
import shutil
import py7zr
import numpy as np
from pandas import DataFrame
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.nn import functional as F

from preprocessing.array import rawseq2array, pad_array, byte_filter, remap
from preprocessing.text import normalize_path
from preprocessing.emulation import emulate
from preprocessing.reports import report_to_apiseq
from utils.structures import report_db, rawpe_db, filepath_db
from utils.functions import sigmoid
from modules.sota.models import MalConvModel, EmberGBDT

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


class Core1DConvNet(nn.Module):
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


class Core1DConvNetAPI(object):
    def __init__(self, device,
                    embedding_dim,
                    vocab_size,
                    state_dict=None, 
                    padding_length=150):
        self.device = device
        self.padding_length = padding_length
        self.model = Core1DConvNet(
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

    def load_state(self, state_dict):
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

                train_loss, train_m = self.train(train_loader, optimizer, loss_function, epoch, verbosity_batches)
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


class Filepath(Core1DConvNetAPI):
    def __init__(self, 
                    bytes, 
                    device, 
                    embedding_dim=64, 
                    state_dict=None,
                    filepath_csv_location=None):
        super().__init__(device, 
                            embedding_dim, 
                            len(bytes)+1, 
                            state_dict)
        self.bytes = bytes
        self.filepath_db = filepath_db(FILEPATH_CSV_LOCATION=filepath_csv_location)

    def predict_path(self, path):
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

    def predict(self, path):
        return self.predict_path(path)


class Emulation(Core1DConvNetAPI):
    def __init__(self, apimap, device, 
                        embedding_dim=96, 
                        state_dict=None, 
                        emulation_report_path=None,
                        speakeasy_config = None):
        super().__init__(device, 
                            embedding_dim, 
                            len(apimap)+2, 
                            state_dict)
        self.apimap = apimap
        self.speakeasy_config = speakeasy_config
        if self.speakeasy_config:
            logging.warning(f"[!] Using speakeasy emulator config from: {self.speakeasy_config}")
        self.report_db = report_db(REPORT_PATH=emulation_report_path)

    def forwardpass_apiseq(self, apiseq):
        x = rawseq2array(apiseq, self.apimap, self.padding_length)
        x = torch.LongTensor(x.reshape(1,-1)).to(self.device)

        self.model.eval()
        logits = self.model(x)
        prediction = torch.argmax(logits, dim=1).flatten()

        return logits, prediction

    def predict_rawpe(self, path, i=0, l=0):
        temp_report_folder = f"temp_reports"
        os.makedirs(temp_report_folder, exist_ok=True)
        
        success = emulate(path, temp_report_folder, speakeasy_config=self.speakeasy_config)
        if not success:
            logging.error(f" [-] Failed emulation of {path}")
            logits = None
        else:
            samplename = os.path.basename(path)
            reportfile = f"{temp_report_folder}/{samplename}.json"
            apiseq = report_to_apiseq(reportfile)["api.seq"]
            logits, _ = self.forwardpass_apiseq(apiseq)
        
        # cleanup
        shutil.rmtree(temp_report_folder)
        return logits, success

    def predict_report(self, h):
        report_fullpath = self.report_db[h]
        apiseq = report_to_apiseq(report_fullpath)["api.seq"]
        logits, preds = self.forwardpass_apiseq(apiseq)
        return logits, preds
    
    def predict(self, h):
        if "/" in h:
            hhash = h.split("/")[-1]
        else:
            hhash = h
        if hhash in self.report_db:
            return self.predict_report(hhash)
        else:
            if os.path.exists(h):
                return self.predict_rawpe(h)
            else:
                raise FileNotFoundError(f"Failed to acquire '{h}'. Is hash or PE path specified correctly?")


class CompositeClassifier(object):
    def __init__(self, modules=["malconv", "ember", "filepaths", "emulation"],
                    root = "./", # repository root
                    
                    # pretrained early fusion components
                    malconv_model_path = 'modules/sota/malconv/parameters/malconv.checkpoint',
                    ember_2019_model_path = 'modules/sota/ember/parameters/ember_model.txt',
                    # metadata used for module pre-training
                    emulation_model_path = 'modules/emulation/pretrained/torch.model',
                    emulation_apicalls = 'modules/emulation/pretrained/pickle.apicalls',
                    speakeasy_config = "data/emulation.dataset/sample_emulation/speakeasy_config.json",
                    filepath_model_path = 'modules/filepath/pretrained/torch.model',
                    filepath_bytes = 'modules/filepath/pretrained/pickle.bytes',
                    # metadata needed for struct instantiation
                    emulation_report_path = "data/emulation.dataset",
                    rawpe_db_path = "data/pe.dataset/PeX86Exe",
                    fielpath_csvs = "data/pe.dataset/PeX86Exe",
                    
                    # later fusion model
                    # options: MultiLayerPerceptron, XGBClassifier, LogisticRegression
                    meta_model = "MultiLayerPerceptron",
                    load_meta_model = True,
                    mlp_hidden_layer_sizes=(15,),
                    
                    # auxiliary settings
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    padding_length = 150
                ):
        # TODO: right now it is really does not accept config w/o preloading models
        # should allow retraining from scratch (all modules, including MalConv/Ember)

        self.module_timers = []
        self.x = None
        self.y = None

        self.device = device
        self.padding_length = padding_length

        # early fusion module actions
        self.root = root
        self.modules = {}

        self.malconv_model_path = self.root + malconv_model_path
        self.ember_2019_model_path = self.root + ember_2019_model_path
        
        if "malconv" in modules:
            self.modules["malconv"] = MalConvModel()
            if os.path.exists(self.malconv_model_path):
                logging.warning(f"[!] Loading pretrained weights for malconv model from: {self.malconv_model_path}")
                self.modules["malconv"].load_state(self.malconv_model_path)
            else:
                malconv_url = "https://github.com/endgameinc/malware_evasion_competition/raw/master/models/malconv/malconv.checkpoint"
                errmsg = f"[-] MalConv pre-trained parameters are missing: {self.malconv_model_path}\nYou can download it from: {malconv_url}"
                logging.error(errmsg)

            
        if "ember" in modules:
            if os.path.exists(self.ember_2019_model_path):
                pass
            else:
                logging.error(f"[-] Ember pre-trained parameters are missing: {self.ember_2019_model_path}")
                if os.path.exists(self.ember_2019_model_path + ".7z"):
                    logging.warning(f"[+] Found archived parameters: {self.ember_2019_model_path + '.7z'}")
                    with py7zr.SevenZipFile(self.ember_2019_model_path + '.7z') as archive:
                        archive.extractall(path=os.path.dirname(self.ember_2019_model_path))
                else:
                    ember_url = "https://github.com/endgameinc/malware_evasion_competition/raw/master/models/ember/ember_model.txt.zip"
                    logging.warning(f"[-] Exclude 'ember' from module list or download parameters from: {ember_url}")
                    sys.exit(1)
            logging.warning(f"[!] Loading pretrained weights for ember model from: {self.ember_2019_model_path}")
            self.modules["ember"] = EmberGBDT(self.ember_2019_model_path)

        if "filepaths" in modules:
            self.bytes = self.root + filepath_bytes
            with open(self.bytes, "rb") as f:
                bytes = pickle.load(f)
            self.modules["filepaths"] = Filepath(bytes, self.device, filepath_csv_location=self.root+fielpath_csvs)
            self.load_filepath_module(self.root + filepath_model_path)

        if "emulation" in modules:
            self.apis = self.root + emulation_apicalls
            with open(self.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            self.apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
            self.modules["emulation"] = Emulation(self.apimap, self.device, 
                                                emulation_report_path=self.root+emulation_report_path,
                                                speakeasy_config=os.path.join(self.root, speakeasy_config))
            self.load_emulation_module(self.root + emulation_model_path)
        
        self.rawpe_db = rawpe_db(self.root+rawpe_db_path)
        
        # late fusion model configuration
        self.meta_model = meta_model
        module_str = '_'.join(sorted(self.modules.keys()))
        if meta_model == "LogisticRegression":
            self.model = LogisticRegression()
            self.meta_model_path = self.root + \
                                    f"modules/meta_model/LogisticRegression_{module_str}.model"

        elif meta_model == "XGBClassifier":
            self.model = XGBClassifier(n_estimators=100, 
                                        objective='binary:logistic',
                                        eval_metric="logloss",
                                        use_label_encoder=False)
            self.meta_model_path = self.root + \
                                    f"modules/meta_model/XGBClassifier_{module_str}.model"

        elif meta_model == "MultiLayerPerceptron":
            self.model = MLPClassifier(hidden_layer_sizes=mlp_hidden_layer_sizes)
            if mlp_hidden_layer_sizes == (15,):
                self.meta_model_path = self.root + \
                                        f"modules/meta_model/MultiLayerPerceptron15_{module_str}.model"
            else:
                self.meta_model_path = None
        else:
            raise NotImplementedError
        
        if load_meta_model:
            if self.meta_model_path and os.path.exists(self.meta_model_path):
                self.load_meta_model(state_path=self.meta_model_path)
            else:
                errmsg = f"[-] No pre-trained late fusion model for this configuration: {self.meta_model_path}. You need to .fit() it!"
                logging.error(errmsg)
                optionlist = [x for x in os.listdir(os.path.dirname(self.meta_model_path)) if x.endswith('.model')]
                modulelist = [x.replace('.model','').split('_')[1:] for x in optionlist]
                modellist = [x.replace('.model','').split('_')[0].replace('15','') for x in optionlist]
                infolist = '\n\t\t'.join(['meta_model=\'{0}\', modules={1}'.format(x,y) for x,y in zip(modellist, modulelist)])
                options = f"[!] Available pre-trained options:\n\t\t{infolist}"
                logging.warning(options)
    
    # late fusion model state actions
    def save_meta_model(self, filename=""):
        filename = filename+f"_{int(time.time())}_"+self.meta_model+".model"
        pickle.dump(self.model, open(filename, 'wb'))
        return filename
        
    def load_meta_model(self, state_path=""):
        msg = f"[!] Loading pretrained weights for late fusion {self.meta_model} model from: {self.meta_model_path}"
        logging.warning(msg)
        self.model = pickle.load(open(state_path, 'rb'))

    # other module actions
    def load_emulation_module(self, state_dict):
        msg = f"[!] Loading pretrained weights for emulation model from: {state_dict}"
        logging.warning(msg)
        self.modules["emulation"].load_state(state_dict)

    def load_filepath_module(self, state_dict):
        msg = f"[!] Loading pretrained weights for filepath model from: {state_dict}"
        logging.warning(msg)
        self.modules["filepaths"].load_state(state_dict)

    # auxiliary
    def dump_xy(self):
        timestamp = int(time.time())
        np.save(f"./X-{timestamp}.npy", self.x)
        logging.warning(f" [!] Dumped module scores to './X-{timestamp}.npy'")
        if self.y:
            np.save(f"./y-{timestamp}.npy", self.y)
            logging.warning(f" [!] Dumped module labels to './y-{timestamp}.npy'")
        
    def get_modular_x(self, modules, x=None):
        modules_all = ["malconv", "ember", "filepaths", "emulation"]
        missing = set(modules_all) - set(modules)
        
        module_indexes = []
        for miss in missing:
            module_indexes.append(modules_all.index(miss))
        
        x_crop = np.delete(x, module_indexes, axis=1)
        return x_crop
    
    def get_module_scores(self, x=None, y=None):
        if x is None:
            if self.x is not None:
                x = self.x
            else:
                logging.error(f"[-] Input array is not defined...")
                return None

        probs = self.predict_proba(x)[:,1]
        x = np.hstack([x, probs.reshape(-1,1)])
        cols = list(self.modules.keys()) + ["Pr(malw.)"]
        
        y = self.y if y is None else y
        if y is not None:
            values = np.hstack([x,y.reshape(-1,1)])
            cols = cols + ["y"]
        else:
            values = x
        
        return DataFrame(values, columns=cols)

    def get_module_processing_time(self):
        return dict(zip(self.modules.keys(), np.vstack(self.module_timers).mean(axis=0)))

    # array actions        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.model.fit(self.x, self.y)
    
    def predict_proba(self, x):
        self.x = x
        return self.model.predict_proba(self.x)

    # PE processing actions
    def _early_fusion_pass(self, pe, filepath=None, defaultpath=None, takepath=False):
        vector = []
        checkpoint = []
        
        # acquire filepath if not provided
        if not filepath and "filepaths" in self.modules:
            if takepath:
                logging.warning(f"[!] Taking current filepath for: {pe}")
                filepath = pe
            elif defaultpath:
                logging.warning(f"[!] Using defaultpath for: {pe}")
                filepath = defaultpath
            else: # get path from filepath db
                if "/" in pe or "\\" in pe: # if fullpaths are given instead filenames
                    pe = os.path.basename(pe)
                if pe in self.modules["filepaths"].filepath_db:
                    filepath = self.modules["filepaths"].filepath_db[pe]
                else: # not in database
                    missing_filepath_error = f"In-the-wild filepath for {pe} is not specified.\n\tAddress using one of the following options in preprocess_pelist():\
\n\t\t a) 'pathlist=' defining in-the-wild filepath for every provided PE file,\n\t\t b) 'defaultpath=' to use the same path for every PE,\
\n\t\t c) 'takepath=True' to use current path on the system,\n\t\t d) remove 'filepaths' from 'modules='."
                    raise ValueError(missing_filepath_error)

        # acquire fullpath of pe if not provided
        if pe in self.rawpe_db:
            pe = self.rawpe_db[pe]
        elif not os.path.exists(pe):
            raise FileNotFoundError(f"Cannot find {pe}, please provide valid path or known hash")

        for model in self.modules:
            # PE modules
            if model == "malconv":
                checkpoint.append(time.time())
                malconv_prob = self.modules["malconv"].get_score(pe)    
                vector.append(malconv_prob)
            
            elif model == "ember":
                checkpoint.append(time.time())
                ember_prob = self.modules["ember"].get_score(pe)
                vector.append(ember_prob)
            
            elif model == "emulation":
                checkpoint.append(time.time())
                emul_logits, success = self.modules["emulation"].predict(pe)
                if success:
                    emul_prob = sigmoid(emul_logits.detach().numpy())[0,1]
                else:
                    # TODO: what to do if emulation fails????
                    emul_prob = 0.5 # np.mean(vector)
                vector.append(emul_prob)
            
            # path module
            elif model == "filepaths":
                checkpoint.append(time.time())
                path_logits, _ = self.modules["filepaths"].predict(filepath)
                path_prob = sigmoid(path_logits.detach().numpy())[0,1]
                vector.append(path_prob)
            
            else:
                raise NotImplementedError            
        
        checkpoint.append(time.time())
        self.module_timers.append([checkpoint[i]-checkpoint[i-1] for i in range(1,len(checkpoint))])
        
        return np.array(vector).reshape(1,-1)
    
    def preprocess_pelist(self, pelist, pathlist=None, defaultpath=None, takepath=False, dump_xy=True):
        x = []
        path = None
        if pathlist and len(pelist) != len(pathlist):
            raise Exception(f"Length of provided pathlist doesn't match length of provided PE file list.")
        
        for i,pe in enumerate(pelist):
            if pathlist:
                path = pathlist[i]
            print(f" [*] Scoring: {i+1}/{len(pelist)}", end="\r")
            x.append(self._early_fusion_pass(pe, path, defaultpath=defaultpath, takepath=takepath))
        
        self.x = np.vstack(x)
        if dump_xy:
            self.dump_xy()

        return self.x

    def fit_pelist(self, pelist, y, pathlist=None, dump_xy=False):
        self.x = self.preprocess_pelist(pelist, pathlist=pathlist)
        self.y = y
        self.model.fit(self.x, self.y)

        if dump_xy:
            self.dump_xy()
    
    def predict_proba_pelist(self, pelist, pathlist=None, dump_xy=False, return_module_scores=False):
        self.x = self.preprocess_pelist(pelist, pathlist=pathlist, dump_xy=dump_xy)
        probs = self.model.predict_proba(self.x)
        
        if return_module_scores:
            return probs, DataFrame(self.x, columns=self.modules.keys())
        else:
            return probs
    
    def update(self, oversampling_rate=100):
        # UPDATE model weights with new data

        # 1. preprocess provided samples

        # 2. oversample

        # 3. form a new dataset out of old samples + new oversamples samples, shuffle
        
        # 4. retrain each module  

        # 5. retrain late fusion model

        # 6. dump new model parameters / new dataset 
        raise NotImplementedError