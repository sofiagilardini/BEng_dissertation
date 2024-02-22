import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import datapipe_redu as dtp
from sklearn.model_selection import train_test_split
import aux_functions as auxf
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin
from scipy.signal import butter, lfilter
import os
import pandas as pd



# winsize = 256ms, stride length = 1ms, cutoff freq = 450Hz

fs = 2000 # 
order = 1
win_size = 256
win_stride = 1
cutoff_f = 450





# dataset 1

dataset = 1

auxf.emg_process(cutoff_val = cutoff_f, size_val = win_size, stride_val = win_stride, user = 1, dataset=dataset, order = order)
auxf.glove_process(size_val = win_size, stride_val = win_stride, user = 1, dataset = dataset)
auxf.restimulusProcess(size_val= win_size, stride_val= win_stride, user = 1, dataset= dataset)


# dataset 2

dataset = 2

auxf.emg_process(cutoff_val = cutoff_f, size_val = win_size, stride_val = win_stride, user = 1, dataset=dataset, order = order)
auxf.glove_process(size_val = win_size, stride_val = win_stride, user = 1, dataset = dataset)
auxf.restimulusProcess(size_val= win_size, stride_val= win_stride, user = 1, dataset= dataset)

#dataset 3 

dataset = 3

auxf.emg_process(cutoff_val = cutoff_f, size_val = win_size, stride_val = win_stride, user = 1, dataset=dataset, order = order)
auxf.glove_process(size_val = win_size, stride_val = win_stride, user = 1, dataset = dataset)
auxf.restimulusProcess(size_val= win_size, stride_val= win_stride, user = 1, dataset= dataset)






# glove = np.load("/home/sofia/beng_thesis/glove_data_processed/glove_1_1_256_1.npy")

# print(glove.shape)

# restimulus_data = np.load("/home/sofia/beng_thesis/restimulus_data_processed/restimulus_1_1_256_1.npy")


# print(restimulus_data.shape)

# restimulus_data = pd.DataFrame(data = restimulus_data)

# isna = restimulus_data.isna()[:1000]

# print(isna)