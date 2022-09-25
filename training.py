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

## Training Loop
def training(model, train_dl, num_epochs, val_dl):
  # Loss Function, Optimizer 
  criterion = nn.CrossEntropyLoss()
  # For SGD
  #optimizer = torch.optim.SGD(model.parameters(), lr= 10**(-5))
  # For Adam
  optimizer = torch.optim.Adam(model.parameters(), lr= 10**(-5))
  
  # For LR Scheduler
  # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3,
  #                                               steps_per_epoch=int(len(train_dl)),
  #                                               epochs=num_epochs,
  #                                               anneal_strategy='linear')
  
  val_accuracy_epoch_list = [] 
  train_accuracy_epoch_list = [] 
  f1_epoch_list = []
  val_precision_list = []
  val_recall_list = []
  best_f1 = 0
  epoch_f1 = 0
  # Epoch iterator
  for epoch in range(num_epochs):
    t0 = time.time()
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    indices = []
    # Batch iterator
    for i, data in enumerate(train_dl):

        inputs, labels = torch.tensor(data[0]).to(device), torch.tensor(data[1]).to(device) # Get the input features and target labels, and put them on the GPU
        if torch.isnan(torch.tensor(data[0])).any() == True:
          continue
        
        # Normalize the inputs
        inputs -= inputs.min(1, keepdim=True)[0]
        inputs /= inputs.max(1, keepdim=True)[0]

        inputs = inputs[None, :, :, :]
        inputs = inputs.permute(1, 0, 2, 3)
        inputs = torch.cat([inputs[:, :, :round(inputs.shape[2]/3), :],inputs[:, :, round(inputs.shape[2]/3):round(inputs.shape[2]*2/3), :],inputs[:, :, round(inputs.shape[2]*2/3):, : ]], dim = 1)
        
        #Following code if we want mean MRCG, shape (256, 6025)
        trans = inputs.detach().cpu().numpy()
        trans =  np.mean(trans, axis = 1)
        inputs = torch.from_numpy(trans).to(device)
        inputs = inputs[None,:,:,:]
       
        optimizer.zero_grad() # Zero the parameter gradients

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # Keep stats for Loss and Accuracy

        _, prediction = torch.max(outputs,1) # Get the predicted class with the highest score
        correct_prediction += (prediction == labels).sum().item() # Count of predictions that matched the target label
        total_prediction += prediction.shape[0]

    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    train_accuracy_epoch_list.append(acc)
    print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    print(f"TESTING:")
    epoch_f1, val_acc, val_prec, val_rec = inference(model, val_dl)
    f1_epoch_list.append(epoch_f1)
    val_accuracy_epoch_list.append(val_acc)
    val_precision_list.append(val_prec)
    val_recall_list.append(val_rec)
    if epoch_f1 > best_f1 and epoch > 9:
      torch.save(model.state_dict(), "/content/gdrive/MyDrive/BestResnet18_ID2.pt")
      best_f1 = epoch_f1
      # model = TheModelClass(*args, **kwargs)
      # model.load_state_dict(torch.load(PATH))
      # model.eval()
    t1 = time.time()
    print(f"time for epoch: {(t1-t0)/60} minutes ")
    torch.save(model.state_dict(), f"/content/gdrive/MyDrive/Resnet18_ID2_Epoch{epoch}.pt")
    print("\n")

  print('Finished Training')

  return train_accuracy_epoch_list, val_accuracy_epoch_list, f1_epoch_list, val_precision_list, val_recall_list


## Inference
def inference(model, val_dl):
  correct_prediction = 0
  total_prediction = 0
  outputsList = []
  labelsList = []
  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalizing
      inputs -= inputs.min(1, keepdim=True)[0]
      inputs /= inputs.max(1, keepdim=True)[0]

      inputs = inputs[None, :, :, :]
      inputs = inputs.permute(1, 0, 2, 3)
      inputs = torch.cat([inputs[:, :, :round(inputs.shape[2]/3), :],inputs[:, :, round(inputs.shape[2]/3):round(inputs.shape[2]*2/3), :],inputs[:, :, round(inputs.shape[2]*2/3):, : ]], dim = 1)

      trans = inputs.detach().cpu().numpy()
      trans =  np.mean(trans, axis = 1)
      inputs =torch.from_numpy(trans).to(device)
      inputs = inputs[None,:,:,:]

      # Get predictions
      outputs = model(inputs.float())

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
      outputsList.append(prediction.cpu().numpy())
      labelsList.append(labels.cpu().numpy())
  # Accuracy
  acc = correct_prediction/total_prediction
  print(f'Val Accuracy: {acc:.2f}')
  # F1 score
  outputs = np.concatenate(outputsList)
  targets = np.concatenate(labelsList)
  precision = Precision(average='micro')
  recall = Recall(average='micro')
  prec = precision(torch.from_numpy(outputs), torch.from_numpy(targets))
  rec = recall(torch.from_numpy(outputs), torch.from_numpy(targets))
  f1 = f1_score(torch.from_numpy(outputs),torch.from_numpy(targets), num_classes = 12, average='micro')
  print(f"F1 score: {f1}")
  return f1, acc, prec, rec