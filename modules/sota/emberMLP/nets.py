from torch import Tensor
from torch import nn
import torch.nn.functional as F

EMBER_FEATURE_DIM = 2381

# NOTE: Ember FFNN from Kyadige and Raff et al.: 
# 1024, 768, 512, 512, 512
class EmberMLP(nn.Module):
    def __init__(self, 
                    input_dim = 2381):
        super().__init__()

        self.fn1 = nn.Linear(input_dim, 1024)
        self.fn2 = nn.Linear(1024, 768)
        self.fn3 = nn.Linear(768, 512)
        self.fn4 = nn.Linear(512, 256)
        self.fn5 = nn.Linear(256, 128)
        self.fn6 = nn.Linear(128, 64)
        self.fn7 = nn.Linear(64, 1)
    
        p = 0.05
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
    
    def forward(self, x):
        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = self.dropout1(x)
        x = F.relu(self.fn4(x))
        x = self.dropout2(x)
        x = F.relu(self.fn5(x))
        x = F.relu(self.fn6(x))
        output = self.fn7(x)
        return output
    
    def get_representations(self, x):
        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = self.dropout1(x)
        x = F.relu(self.fn4(x))
        x = self.dropout2(x)
        representations = F.relu(self.fn5(x))
        return representations

    
class BasicMLP(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


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
        
    def forward(self,data):
        base_result = self.model_base(data)
        return self.malware_head(base_result)
    
    def get_representations(self, x):
        return self.model_base(Tensor(x)).reshape(-1,1)


# from: https://github.com/sophos-ai/SOREL-20M/blob/master/nets.py
# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.
class PENetworkOrig(nn.Module):
    """
    This is a simple network loosely based on the one used in ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation (https://arxiv.org/abs/1903.05700)
    Note that it uses fewer (and smaller) layers, as well as a single layer for all tag predictions, performance will suffer accordingly.
    """
    def __init__(self,use_malware=True,use_counts=True,use_tags=True,n_tags=11,feature_dimension=1024, layer_sizes = None):
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        if self.use_tags and self.n_tags == None:
            raise ValueError("n_tags was None but we're trying to predict tags. Please include n_tags")
        super(PENetworkOrig,self).__init__()
        p = 0.05
        layers = []
        if layer_sizes is None:
            layer_sizes=[512,512,128]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(feature_dimension,ls))
            else:
                layers.append(nn.Linear(layer_sizes[i-1],ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p))
        self.model_base = nn.Sequential(*tuple(layers))
        self.malware_head = nn.Sequential(nn.Linear(layer_sizes[-1], 1),
                                          nn.Sigmoid())
        self.count_head = nn.Linear(layer_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.tag_head = nn.Sequential(nn.Linear(layer_sizes[-1],64),
                                        nn.ELU(), 
                                        nn.Linear(64,64),
                                        nn.ELU(),
                                        nn.Linear(64,n_tags),
                                        nn.Sigmoid())

    def forward(self,data):
        base_result = self.model_base(data)
        return self.malware_head(base_result)
    
    def get_representations(self, x):
        return self.model_base(Tensor(x)).reshape(-1,1)

