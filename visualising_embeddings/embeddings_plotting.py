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


gestures = np.arange(1, 10, 1)

dim_list = [2, 3]
user_list = [1, 2, 3, 6, 7]



for user in user_list:


    emg_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'emg')
    emg_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'emg')
    emg_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'emg')

    glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'glove')
    glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'glove')
    glove_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'glove')

    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'restimulus').astype(int)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'restimulus').astype(int)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)



    for dim in dim_list:

        if dim == 3:
            fig, axes = plt.subplots(3, 3, figsize=(20, 20), subplot_kw={'projection': '3d'})  # All subplots are 3D

        if dim == 2:
            fig, axes = plt.subplots(3, 3, figsize=(20, 20))  


        iterations = 10000


        cebra_model = CEBRA(
            model_architecture = 'offset10-model',
            batch_size= 64,
            temperature_mode='auto',
            learning_rate = 0.0001,
            max_iterations = iterations,
            time_offsets = 25,
            output_dimension = dim, 
            device = "cuda_if_available",
            verbose = True,
            conditional='time_delta',
            distance = 'cosine',
        )

        cebra_model.partial_fit(emg_tr1, glove_tr1, restim_tr1)
        cebra_model.partial_fit(emg_tr2, glove_tr2, restim_tr1)

        cebra_model.save(f"./visualising_embeddings/models/user{user}_dim{dim}.pt")
        cebra.plot_loss(cebra_model)
        plt.savefig(f"./visualising_embeddings/models/user{user}_dim{dim}_loss.png")


        axes = axes.flatten()

        for i, ax in enumerate(axes):

            print("i", i)

            gesture = gestures[i]
            print("gesture", gesture)

            start, end = auxf.cutStimTransition(restim_test, gesture)
            num_channels_emg = emg_tr1.shape[1]

            embedding = cebra_model.transform(emg_test[start:end])

            print('embedding shape', embedding.shape)


            # Plotting the embedding on each subplot
            cebra.plot_embedding(embedding, cmap = "magma", markersize=5, alpha = 0.5, embedding_labels='time', ax = ax, title = f"Gesture {gesture}, User {user}")

            # Customizing each subplot
            ax.set_facecolor('black')

            xticks = np.arange(-1, 1, 0.5)


            if dim == 3:
                ax.set_title(f"Gesture {gesture}, User {user}" , color = 'white', fontsize = 15)
                ax.tick_params(axis='x', colors='white', size = 5)
                ax.tick_params(axis='y', colors='white', size = 5)
                ax.tick_params(axis='z', colors='white', size = 5)
                # ax.set_xticks(xticks)
                # ax.set_yticks(xticks)



            if dim == 2:
                ax.set_title(f"Gesture {gesture}, User {user}" , color = 'black', fontsize = 15)
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                # ax.set_xticks(xticks)




        fig.savefig(f"./visualising_embeddings/embeddings/user{user}_dim{dim}.png")

    
