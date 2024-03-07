import sys

# print the original sys.path
print('Original sys.path:', sys.path)
sys.path.append("/home/sofia/beng_thesis")
print("updated", sys.path)


import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
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
from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin
from copy import deepcopy
import seaborn as sns
import os

from sklearn.decomposition import PCA



restim = auxf.getProcessedData(user = 1, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)

emg = auxf.getProcessedData(user = 1, dataset = 3, type_data = 'test', mode = 'emg')

glove = auxf.getProcessedData(user = 1, dataset = 3, type_data = 'test', mode = 'glove')

gesture = 3


start, end = auxf.cutStimTransition(restim, gesture)
num_channels_emg = emg.shape[1]


LV_channels = np.arange(0, num_channels_emg, 2)
WL_channels = np.arange(1, num_channels_emg, 2)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))

# First subplot for restim data
ax1.set_title(f"Restimulus Data - Transition from {gesture} to {gesture+1}")
ax1.set_xlabel("Windows")
ax1.set_ylabel("Gesture")

ax1.plot(np.arange(start, end, 1), restim[start: end], 'black')
ax1.set_yticks([0, gesture, gesture + 1])


# Second subplot for emg data
ax2.set_title("EMG Log Variance - Corresponding Segment")
ax2.set_xlabel("Windows")
ax2.set_ylabel("Log Variance")
ax2.plot(np.arange(start, end, 1), emg[start: end, LV_channels])

# Second subplot for emg data
ax3.set_title("EMG Waveform Length - Corresponding Segment")
ax3.set_xlabel("Windows")
ax3.set_ylabel("Waveform Length")
ax3.plot(np.arange(start, end, 1), emg[start: end, WL_channels])


ax4.set_title("Glove Data - Corresponding Segment")
ax4.set_xlabel("Windows")
ax4.set_ylabel("Glove readings")
ax4.plot(np.arange(start, end, 1), glove[start: end])


# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"./visualising_embeddings/images_saved/transition_stimulus_emg_{gesture}_{gesture+1}.png")

plt.show()
