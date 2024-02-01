import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import datapipe as dtp
from sklearn.model_selection import train_test_split
import aux_functions as auxf
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin
from scipy.signal import butter, lfilter


# user is S1-S12 and dataset is A1-A3 (note A3 only has 2 repetitions of each movement)
def getdata(user, dataset):
    data = scipy.io.loadmat(f"./datasets/S{user}_E1_A{dataset}.mat")
    emg_data = data['emg']
    glove_data = data['glove']
    stimulus_data = data['stimulus']

    return emg_data, glove_data, stimulus_data


# I want a function that takes in my desired channels for glove data
# and returns a list of lists with only those channels ? do I ? 

# 1. Define the parameters, either variable or fixed
params_grid = dict(
    model_architecture = "offset5-model",
    batch_size = [2**4, 2**5, 2**6, 2**7],
    output_dimension = [3, 6, 9, 12],
    learning_rate = [0.001, 0.005, 0.0001],
    time_offsets = [1,2,5,10,15,20],
    max_iterations = 10000,
    temperature_mode = "auto",
    verbose = True)

