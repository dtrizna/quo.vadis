import os
import sys

root = "../../"
sys.path.append(root)
from models import Filepath, Emulation

import torch
from torch import nn

from utils.structures import rawpe_db
from ember import PEFeatureExtractor
import logging
import pickle
import time

import numpy as np
from pandas import DataFrame

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

EMBER_FEATURE_DIM = 2381

# from: https://github.com/sophos-ai/SOREL-20M/blob/master/nets.py
# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.
class PENetwork(nn.Module):
    """
    This is a simple network loosely based on the one used in ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation (https://arxiv.org/abs/1903.05700)
    Note that it uses fewer (and smaller) layers, as well as a single layer for all tag predictions, performance will suffer accordingly.
    """
    def __init__(self, feature_dimension=EMBER_FEATURE_DIM, layer_sizes = None):
        super(PENetwork,self).__init__()
        p = 0.05
        layers = []
        if layer_sizes is None:
            layer_sizes = [512, 512, 128]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(feature_dimension, ls))
            else:
                layers.append(nn.Linear(layer_sizes[i-1], ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p))
        
        self.model_base = nn.Sequential(*tuple(layers))
        
        self.malware_head = nn.Sequential(nn.Linear(layer_sizes[-1], 1))
        
    def forward(self, data):
        base_result = self.model_base(data)
        return self.malware_head(base_result)
    
    def get_representations(self, x):
        return self.model_base(torch.Tensor(x)).reshape(-1,1)
    
    def classify_representations(self, representations):
        return self.malware_head(representations)


class CompositeClassifierFromRepresentations(object):
    def __init__(self, modules=["malconv", "ember", "filepaths", "emulation"],
                    root = "./", # repository root
                    
                    # pretrained early fusion components
                    emulation_model_path = 'modules/emulation/pretrained/torch.model',
                    filepath_model_path = 'modules/filepath/pretrained/torch.model',
                    ember_mlp_model_path = 'modules/sota/emberMLP/model_myPENetwork_190_epochs_acc_90/ep5-optim_adamw-lr0.001-l2reg0-dr0.5-model.torch',
                    
                    # metadata objects used for pre-training model functionality (mandatory)
                    emulation_apicalls = 'modules/emulation/pretrained/pickle.apicalls',
                    speakeasy_config = "data/emulation.dataset/sample_emulation/speakeasy_config.json",
                    filepath_bytes = 'modules/filepath/pretrained/pickle.bytes',

                    # metadata needed for struct instantiation (optional)
                    emulation_report_path = "data/emulation.dataset/tainvalset_emulaiton",
                    rawpe_db_path = "data/archives/pe_trainset/PeX86Exe",
                    fielpath_csvs = "data/archives/pe_trainset/PeX86Exe",
                    
                    # meta model -- options: MultiLayerPerceptron, XGBClassifier, LogisticRegression
                    meta_model = "MultiLayerPerceptron",
                    load_meta_model = True,
                    mlp_hidden_layer_sizes=(128,64,32), # only needed if meta_model="MultiLayerPerceptron"
                    meta_fit_max_iter=200, # only needed if meta_model="MultiLayerPerceptron"

                    # auxiliary settings
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    padding_length = 150
                ):
        self.module_timers = []
        self.x = None
        self.y = None

        self.device = device
        self.padding_length = padding_length

        # early fusion module actions
        self.root = root
        self.modules = {}

        self.ember_mlp_model_path = self.root + ember_mlp_model_path

        if "ember" in modules:
            self.modules["ember"] = PENetwork()
            if os.path.exists(self.ember_mlp_model_path):
                logging.warning(f"[!] Loading pretrained weights for ember model from: {self.ember_mlp_model_path}")
                self.modules["ember"].load_state_dict(torch.load(self.ember_mlp_model_path))
            else:
                logging.error(f"[-] {self.ember_mlp_model_path} doesn't exist...")
        
        if "filepaths" in modules:
            self.bytes = self.root + filepath_bytes
            self.fielpath_csvs = self.root + fielpath_csvs
            with open(self.bytes, "rb") as f:
                bytes = pickle.load(f)
            self.modules["filepaths"] = Filepath(bytes, self.device, 
                                                    get_representations=True,
                                                    filepath_csv_location=self.fielpath_csvs
                                                )
            self.load_filepath_module(self.root + filepath_model_path)

        if "emulation" in modules:
            self.apis = self.root + emulation_apicalls
            self.emulation_report_path = self.root + emulation_report_path
            self.speakeasy_config = os.path.join(self.root, speakeasy_config)
            with open(self.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            self.apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
            self.modules["emulation"] = Emulation(self.apimap, self.device, 
                                                    get_representations=True,
                                                    emulation_report_path=self.emulation_report_path,
                                                    speakeasy_config=self.speakeasy_config
                                                )
            self.load_emulation_module(self.root + emulation_model_path)
        
        self.rawpe_db = rawpe_db(self.root+rawpe_db_path)
        
        # late fusion model configuration
        self.meta_model = meta_model
        module_str = '_'.join(sorted(self.modules.keys()))
        
        if meta_model == "LogisticRegression":
            self.model = LogisticRegression()
        elif meta_model == "XGBClassifier":
            self.model = XGBClassifier(n_estimators=100, 
                                        objective='binary:logistic',
                                        eval_metric="logloss",
                                        use_label_encoder=False)
        elif meta_model == "MultiLayerPerceptron":
            self.model = MLPClassifier(hidden_layer_sizes=mlp_hidden_layer_sizes,
                                        max_iter=meta_fit_max_iter,
                                        random_state=42) # TBD
        else:
            raise NotImplementedError
    
    
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
    
    def get_module_scores(self, x=None, y=None):
        # TODO: needs fix: 
        # for model in self.modules:
        #   For emberMLP:  
        #   self.modules[model].malware_head(torch.Tensor(x[:5,0:128])).squeeze()
        #   [OUT] tensor([0.2675, 0.2100, 0.3408, 0.0151, 0.2808], grad_fn=<SqueezeBackward0>)
        raise NotImplementedError

    def _build_module_indice_map(self):
        # TODO:
        # last_linear_layer = [x for x in model.modules["emulation"].model.ffnn[-1] if x._get_name() == "Linear"]
        # how many neurons in representation: last_linear_layer[0].out_features
        return dict(zip(self.modules.keys(), ([0, 128], [128, 128*2], [128*2, 128*3])))

    def get_modular_x(self, modules, x):
        module_map = self._build_module_indice_map()
        modular_x = []
        for module in self.modules: # to preserve sequence
            if module in modules:
                idxs = module_map[module]
                modular_x.append(x[:,idxs[0]:idxs[1]])
        return np.hstack(modular_x)


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
            #raise FileNotFoundError(f"Cannot find {pe}, please provide valid path or known hash")
            logging.error(f"[-] Cannot find {pe}, please provide valid path or known hash. Skipping...")
            return None

        # actual pass
        for model in self.modules:
            
            if model == "ember":
                checkpoint.append(time.time())
                extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
                with open(pe, 'rb') as fp:
                    bytez = fp.read()
                ember_features = np.array(extractor.feature_vector(bytez), dtype=np.float32)
                ember_representations = self.modules["ember"].get_representations(ember_features)
                vector.append(ember_representations)
                        
            elif model == "filepaths":
                checkpoint.append(time.time())
                path_representations, _ = self.modules["filepaths"].predict(filepath)
                vector.append(path_representations)

            elif model == "emulation":
                checkpoint.append(time.time())
                emul_representations, success = self.modules["emulation"].predict(pe)
                if success:
                    pass
                else:
                    # TODO: what to do if emulation fails????
                    # load meta-model w/o emulation in modules???
                    print(f"[-] Failed emulation: {pe}...")
                    emul_representations = torch.empty(size=(128,1))
                vector.append(emul_representations)
            
            else:
                raise NotImplementedError            
        
        checkpoint.append(time.time())
        self.module_timers.append([checkpoint[i]-checkpoint[i-1] for i in range(1,len(checkpoint))])
        
        return torch.vstack(vector).reshape(1,-1).detach().numpy()
        
    def preprocess_pelist(self, pelist, pathlist=None, defaultpath=None, takepath=False, dump_xy=True):
        x = []
        path = None
        if pathlist and len(pelist) != len(pathlist):
            raise Exception(f"Length of provided pathlist doesn't match length of provided PE file list.")
        
        for i,pe in enumerate(pelist):
            if pathlist:
                path = pathlist[i]
            print(f" [*] Scoring: {i+1}/{len(pelist)}", end="\r")
            vector = self._early_fusion_pass(pe, path, defaultpath=defaultpath, takepath=takepath)
            if vector is not None:
                x.append(vector)

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