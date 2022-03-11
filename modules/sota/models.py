import torch
import torch.nn.functional as F
import lightgbm as lgb
import numpy as np
import subprocess
import sys
sys.path.append("/data/quo.vadis/modules/sota")
from ember import predict_sample
from malconv import MalConv

class MalConvModel(object):
    def __init__(self, model_path, thresh=0.5, name='malconv'): 
        self.model = MalConv(channels=256, window_size=512, embd_size=8).train()
        weights = torch.load(model_path,map_location='cpu')
        self.model.load_state_dict( weights['model_state_dict'])
        self.thresh = thresh
        self.__name__ = name

    def get_score(self, file_path):
        try:
            with open(file_path, 'rb') as fp:
                bytez = fp.read(2000000)        # read the first 2000000 bytes
                _inp = torch.from_numpy( np.frombuffer(bytez, dtype=np.uint8)[np.newaxis,:].copy() )
                with torch.no_grad():
                    outputs = F.softmax( self.model(_inp), dim=-1)
                return outputs.detach().numpy()[0,1]
        except Exception as e:
            print(e)
        return 0.0 
    
    def is_evasive(self, file_path):
        score = self.get_score(file_path)
        #print(os.path.basename(file_path), score)
        return score < self.thresh


class EmberModel_2019(object):       # model in MLSEC 2019
    def __init__(self, model_path, thresh=0.8336, name='ember'):
        # load lightgbm model
        self.model = lgb.Booster(model_file=model_path)
        self.thresh = thresh
        self.__name__ = 'ember'

    def get_score(self,file_path):
        with open(file_path, 'rb') as fp:
            bytez = fp.read()
            score = predict_sample(self.model, bytez)
            return score
    
    def is_evasive(self, file_path):
        score = self.get_score(file_path)
        return score < self.thresh


class ClamAV(object):
    def is_evasive(self, file_path):
        res = subprocess.run(['clamdscan', '--fdpass', file_path], stdout=subprocess.PIPE)
        #print(res.stdout)
        if 'FOUND' in str(res.stdout):
            return False
        elif 'OK' in str(res.stdout):
            return True
        else:
            print('clamav error')
            exit()
