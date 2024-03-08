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

dim_list = [3]
user_list = [7, 6, 3, 1, 2]
# user_list = [7, 6, 3, 2, 1]



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
            continue


        iterations = 10000


        cebra_model = CEBRA(
            model_architecture = 'offset10-model',
            batch_size= 256,
            temperature_mode='auto',
            min_temperature= 1.2,
            learning_rate = 0.0001,
            max_iterations = iterations,
            time_offsets = 25,
            output_dimension = dim, 
            device = "cuda_if_available",
            verbose = True,
            conditional='time_delta',
            distance = 'cosine' 
        )   


        #glove_channels = [1, 4, 6, 8, 11, 14]

        glove_channels = [0, 3, 5, 7, 10, 13] # because 0 index


        # cebra_model.partial_fit(emg_tr1, glove_tr1[: , glove_channels])
        # cebra_model.partial_fit(emg_tr2, glove_tr2[:, glove_channels])

        location = "models_glove_channelscut"

        # cebra_model.save(f"./visualising_embeddings/{location}/models/user{user}_dim{dim}.pt")
        # cebra.plot_loss(cebra_model)
        # plt.savefig(f"./visualising_embeddings/{location}/models/user{user}_dim{dim}_loss.png")
        cebra_model = cebra.CEBRA.load(f"./visualising_embeddings/{location}/models/user{user}_dim{dim}.pt")


        axes = axes.flatten()

        for i, ax in enumerate(axes):

            print("i", i)

            gesture = gestures[i]
            print("gesture", gesture)


            start, end = auxf.cutStimTransition(restim_test, gesture)
            print(f"on gesture {gesture}, start {start}, end {end}")
            num_channels_emg = emg_tr1.shape[1]


            if gesture != 9:
                embedding = cebra_model.transform(emg_test[start:(end)])
                labels = restim_test[start: (end)]
            else: 
                embedding = cebra_model.transform(emg_test[start:(end)])
                labels = restim_test[start: (end)]


            print('embedding shape', embedding.shape)

            labels = labels.flatten()
            print(labels)




            # # Plotting the embedding on each subplot
            cebra.plot_embedding(embedding, cmap = 'magma', markersize=7, alpha = 0.8, embedding_labels= 'time', ax = ax, title = f"Gesture {gesture}, User {user}")




            # # Generate a unique color for each label
            # unique_labels = np.unique(labels)
            # colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
            # label_color_dict = dict(zip(unique_labels, colors))
            

            
            # # all_labels = [10, 7, ]
            all_labels = [0, 1, 2, 3, 4,5, 6, 7, 8, 9 ,10]

            unique_labels = np.unique(all_labels)
            colors = plt.cm.jet(np.linspace(0, 1, len(all_labels)))

            #Create a mapping from labels to colors
            color_map = dict(zip(unique_labels, colors))

            # all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            # # Unique labels from your data
            # unique_labels = np.unique(all_labels)

            # # Initialize an empty dictionary for color mapping
            # color_map = {}

            # # Assign black to label '0'
            # color_map[0] = 'white'

            # # Use a colormap for the rest of the labels
            # colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels) - 1))  # We subtract 1 because '0' is already assigned

            # # Assign the rest of the colors
            # for i, label in enumerate(unique_labels):
            #     if label != 0:
            #         color_map[label] = colors[i - 1]  # Subtract 1 to account for the black color already assigned


            # Assign colors to each data point
            point_colors = [color_map[label] for label in labels]


            # Customizing each subplot
            ax.set_facecolor('dimgray')

            xticks = np.arange(-1, 1, 0.5)


            from matplotlib.patches import Patch
            legend_handles = [Patch(color=color_map[label], label=label) for label in unique_labels]


            if dim == 3:
                #ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 7, alpha = 0.8)
                ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'black', fontsize = 15)
                ax.tick_params(axis='x', colors='white', size = 5)
                ax.tick_params(axis='y', colors='white', size = 5)
                ax.tick_params(axis='z', colors='white', size = 5)

                #ax.legend(handles=legend_handles)
                #ax.legend(labels)
                # ax.set_xticks(xticks)
                # ax.set_yticks(xticks)
                # for i, label in enumerate(labels):
                #     x, y, z = embedding[i]
                #     ax.text(x, y, z, int(label))





            if dim == 2:
                continue
                ax.scatter(embedding[:, 0], embedding[:, 1], c = point_colors, s = 7, alpha = 0.8)
                ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'black', fontsize = 15)
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                #ax.legend(handles=legend_handles)

                #ax.legend(labels)
                # ax.set_xticks(xticks)
                # for i, label in enumerate(labels):
                #     x, y,= embedding[i]
                #     ax.text(x, y, int(label))

                

    
        fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1))  # You can adjust the location as needed


        plt.show()

        #fig.savefig(f"./visualising_embeddings/{location}/embeddings/user{user}_dim{dim}.png")

    
