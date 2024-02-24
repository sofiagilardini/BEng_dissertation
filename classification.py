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


emg_data = np.load("/home/sofia/beng_thesis/validation_data/emg_data_processed/emg_1_2_450_256_1_0102.npy")
glove_data = np.load("/home/sofia/beng_thesis/validation_data/glove_data_processed/glove_1_2_256_1.npy")
restimulus_data = np.load("/home/sofia/beng_thesis/validation_data/restimulus_data_processed/restimulus_1_2_256_1.npy")

# restimulus_data = pd.DataFrame(data = glove_data)

# isna = restimulus_data.isna()

# print(isna)

# emg_train, emg_test, glove_train, glove_test, rest_train, rest_test = train_test_split(emg_data, 
#                                                                                        glove_data, 
#                                                                                        restimulus_data, 
#                                                                                        train_size=0.7)

from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin

@cebra.models.register("my-model") 
class MyModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 40),
            nn.GELU(),
            nn.Conv1d(num_units, num_output, 5),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(22, 23)


# define cebra model
cebra_model1 = CEBRA(
    model_architecture = "offset10-model",
    batch_size = 300,
    temperature_mode='auto',
    #temperature= 0.9,
    learning_rate = 0.01,
    max_iterations = 1000,
    time_offsets = 1,
    output_dimension = 6,
    device = "cuda_if_available",
    verbose = True,
    conditional='time_delta',
    distance = 'cosine',
)

test = 9000

train = 80000

emg_train = emg_data[:train, :]
emg_test = emg_data[train:(train+test), :]
glove_train = glove_data[:train, :]
restimulus_train = restimulus_data[:train, :]


cebra_model1.fit(emg_train, glove_train, restimulus_train)

embedding = cebra_model1.transform(emg_test)

#accracy = auxf.cKNN(cebra_model1, emg_u1_d1_train, emg_u1_d1_test, glove_u1_d1_train, glove_u1_d1_test)
#print(accracy)

auxf.plotEmbedding(cebra_model1, embedding)