"""
This file is to encode SMILES and SELFIES into one-hot encodings
"""

import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Int_selfies_dataset(Dataset):

    def __init__(self,npy_path,json_path) :
        """
        Args:
            npy_path (string): Path to the npy file with the selfies encoded as integers.
            json_path (string): Path to the jsonfile with the selfies alphabets and  encoded as integers.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.selfies_integers=torch.tensor(np.load(npy_path))
        with open(json_path) as json_file:
            json_dict = json.load(json_file)
        self.alphabet=json_dict['selfies_alphabet']
        self.largest_selfie_len=json_dict['largest_selfies_len']
    
    def __len__(self):
        return len(self.selfies_integers)
    
    def __getitem__(self, idx):
        # one_hot_selfie=F.one_hot(self.selfies_integers[idx])
        one_hot_selfie=F.one_hot(self.selfies_integers[idx],num_classes=len(self.alphabet)).float() 
        return one_hot_selfie
    
    def vae_param(self):
        alphabet_len=len(self.alphabet)
        return alphabet_len,self.largest_selfie_len,self.alphabet



