import torch
from torch import nn
from skorch import NeuralNetRegressor

import sys
sys.path.append("/home/sofia/BEng_diss")


import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import aux_functions as auxf
from copy import deepcopy
from sklearn.metrics import accuracy_score
import cebra.models
import cebra.data
import os
import pandas as pd
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from skorch import NeuralNetRegressor

from joblib import dump
from joblib import load 
from skorch.callbacks import EarlyStopping
import random

# Seed value
seed_value = 42

# Python's built-in random module
random.seed(seed_value)

# Numpy's random number generator
np.random.seed(seed_value)

# PyTorch's random number generator
torch.manual_seed(seed_value)

# CUDA's random number generator (if using GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU.

# Additional settings for PyTorch to further ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='lstm', use_mlp=False):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.use_mlp = use_mlp
        if self.use_mlp:
                # Choose the type of RNN
            if rnn_type.lower() == 'lstm':
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            elif rnn_type.lower() == 'gru':
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
            else:  # Default to simple RNN
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
            
            # Define the output layer
            self.linear = nn.Linear(hidden_size, output_size)

            # Define input MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )

        else:
            # Choose the type of RNN
            if rnn_type.lower() == 'lstm':
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            elif rnn_type.lower() == 'gru':
                self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            else:  # Default to simple RNN
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            
            # Define the output layer
            self.linear = nn.Linear(hidden_size, output_size)
            self.mlp = None
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if self.mlp is not None and self.use_mlp:
            x = self.mlp(x)
        if hasattr(self, 'rnn') and isinstance(self.rnn, nn.LSTM):
            # If LSTM, also initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        # out = out[:, :, -1]
        out = self.linear(out[:, -1, :])
        return out
    



user = 7

emg1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
emg2 = auxf.getProcessedEMG(user = user, dataset= 2, type_data = 'all')

stim1 = auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
stim2 = auxf.getProcessedData(user = user, dataset=2, mode = 'restimulus', rawBool=False)

glove1 = (auxf.getMappedGlove(user = user, dataset=1)).T
glove2 = (auxf.getMappedGlove(user = user, dataset=2)).T

# create trainval
emg_trainval = np.concatenate((emg1, emg2))
stim_trainval = np.concatenate((stim1, stim2))
glove_trainval = np.concatenate((glove1, glove2))




scaler = MinMaxScaler(feature_range=(0, 1))
glove_trainval = scaler.fit_transform(glove_trainval)



print("finished loading")


cebra_model = CEBRA(
    model_architecture = "offset10-model",
    batch_size = 256,
    temperature_mode="auto",
    learning_rate = 0.0001,
    max_iterations = 20000,
    time_offsets = 25,
    output_dimension = 3,
    device = "cuda_if_available",
    verbose = True,
    conditional='time_delta',
)

cebra_model.fit(emg_trainval, glove_trainval)


cebra.CEBRA.save(cebra_model, './cebra_model_test.pt')

cebra_model = cebra.CEBRA.load('./cebra_model_test.pt')


emg_trainval_cebra = cebra_model.transform(emg_trainval)


scaler = MinMaxScaler(feature_range=(0, 1))
emg_trainval_cebra = scaler.fit_transform(emg_trainval_cebra)

scaler = MinMaxScaler(feature_range=(0, 1))
emg_trainval = scaler.fit_transform(emg_trainval)



def create_sequences2(X, y, seq_len):
    X_seq, y_seq = [], []
    
    assert len(X) == len(y)
    for i in range(len(X) - seq_len):
        X_seq_i = X[i:i+seq_len]
        y_i = y[i+seq_len]
        X_seq.append(X_seq_i)
        y_seq.append(y_i)
    
        if i%1000 == 0:
            print("i", i)
    
    print('finished y fer is a potato :)')

    X_torch =  torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_torch = torch.tensor(np.array(y_seq), dtype=torch.float32)

    print('finished TORCH fer is a potato :)')


    return X_torch, y_torch




# # Example usage
# seq_len = 10  # Choose a sequence length
# emg1_seq = create_sequences(emg1, seq_len)
# glove1_seq = glove1[seq_len-1:]  # Align glove data with the sequences ???? 
# # glove1_seq = torch.tensor(glove1[seq_len:], dtype=torch.float32)  # Convert glove data




emg1_seq, glove1_seq = create_sequences2(emg_trainval_cebra, glove_trainval, 2)


# increase seq length 
# incase layers 
# increase hidden size 

# Hyperparameters
input_size = 3  # The input dimension
output_size = 5  # The output dimension
hidden_size = 6  # The number of features in the hidden state
num_layers = 1 # The number of recurrent layers

early_stopping = EarlyStopping(
    monitor='valid_loss',     # Metric to monitor
    lower_is_better=True,     # Set to `True` because we want to minimize loss
    patience=5,               # Number of epochs to wait after last time validation loss improved
    threshold=0.00001,         # Minimum change to qualify as an improvement
    threshold_mode='rel',     # 'rel' means relative change, 'abs' means absolute change
)

# Initialize the RNN model with an LSTM (you can change rnn_type to 'lstm' or 'gru' or 'rnn')
model = RNNRegressor(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, rnn_type='gru')

# Wrap the model with skorch
net = NeuralNetRegressor(
    model,
    criterion=torch.nn.MSELoss,  # Mean Squared Error Loss for regression
    optimizer=torch.optim.Adam,  # Adam optimizer
    lr=0.00001,  # Learning rate
    batch_size=2**6,  # Batch size for training
    max_epochs=2000,  # Number of epochs to train
    device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    verbose = True,
    callbacks=[early_stopping]  # Add the early stopping callback here

)

# Train the RNN with the generated sinusoidal data
net.fit(emg1_seq, glove1_seq)

dump(net, './nettest.joblib')

# After training, generate predictions
predictions = net.predict(emg1_seq)


# Plotting the first n samples for visualization
n_samples_to_plot = 2000  # Set this to the number of samples you want to plot

plt.figure(figsize=(15, 6))
for i in range(output_size):  # Assuming output_size is the number of features in glove data
    plt.subplot(output_size, 1, i+1)
    plt.plot(glove1_seq[:n_samples_to_plot, i], label='Actual')
    plt.plot(predictions[:n_samples_to_plot, i], label='Predicted', alpha=0.7)
    plt.title(f'Feature {i+1}')
    plt.legend()

plt.tight_layout()

plt.show()

score = r2_score(glove1_seq, predictions)

print("score", score)