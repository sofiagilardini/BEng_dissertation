# import pandas as pd
# import scipy as scipy
# import numpy as np
# import matplotlib.pyplot as plt

# data = scipy.io.loadmat("./datasets/S1_E1_A1.mat")
# keys = data.keys()

# # dict_keys(['__header__', '__version__', '__globals__', 'subject', 'exercise', 'emg', 'acc', 'gyro', 'mag', 'glove', 'stimulus', 'repetition', 'restimulus', 'rerepetition'])
# print(keys)

# # this comes out as numpy.ndarray
# print(type(data['emg']))
# print((data['emg'].shape)) # = (2292526, 16)

# # this comes out as numpy.ndarray
# print(type(data['emg'][0]))
# print((data['emg'][0].shape)) # = (16,) --- there are 16 emg channels

# # this comes out as numpy.float32

# print(type(data['emg'][0][0]))

# testarray = data["emg"]

# print(testarray.shape)

# testarray_oneemg = testarray[:, 2]
# testarray_twoemg = testarray[:, 1]

# plt.plot(testarray_oneemg, 'r')
# plt.plot(testarray_twoemg, 'b')
# plt.show()



import matplotlib.pyplot as plt
import scipy.io 
import numpy as np 

data = scipy.io.loadmat("./datasets/S1_E1_A1.mat")

emg_data = data['emg']

num_emg_channels = emg_data.shape[1]

# subplots
plots_columns = 4
plots_rows = 4

# Create a figure and an array of subplots
fig, axs = plt.subplots(plots_rows, plots_columns, figsize=(10, 8))

channel = 0
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'lime', 'pink', 'brown', 'teal', 'gray', 'olive', 'navy']

# Loop through the subplots and plot the second column in each
for row in range(plots_rows):
    for col in range(plots_columns):

        #i = row * plots_columns + col  # Index for accessing the emg column
        axs[row, col].plot(emg_data[:, channel], label=f'Channel {channel + 1}', color = 'k')
        axs[row, col].set_title(f'EMG Channel {channel + 1}')
        axs[row, col].legend()
        channel += 1

plt.tight_layout()
plt.show()


