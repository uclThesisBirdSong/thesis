import zipfile
import os
import pandas as pd
import math, random
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from sklearn.model_selection import StratifiedShuffleSplit
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft
from scipy.signal import lfilter
import scipy.io as sio
from scipy import signal
import gc
import h5py
from torchsummary import summary
!pip install torchmetrics
from torchmetrics import Precision, Recall, ConfusionMatrix
from torchmetrics.functional import f1_score




## Data Loader for MRCG
class H5DS(Dataset):
  def __init__(self, df, path, toy_ind):
    self.path = path
    self.data = h5py.File(self.path, 'r')['data']
    self.df = df
    self.toy_ind = toy_ind
  
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
   
   return (self.data[toy_ind[idx]], torch.tensor(self.df['primary_label'].iloc[idx]))





## Data Loader for Spectrograms
class MelSpecDS(Dataset):
  def __init__(self, df):
    self.df = df
  
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
   spec = get_mel_spec_from_file(self.df['relative_path'].iloc[idx])
   return (spec, torch.tensor(self.df['primary_label'].iloc[idx]))