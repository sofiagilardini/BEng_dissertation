import sys

# print the original sys.path
print('Original sys.path:', sys.path)
sys.path.append("/home/sofia/beng_thesis")
print("updated", sys.path)


import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
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
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch





test = False


user_list = [1, 2, 3, 4, 6, 7, 10]
emg_types = ['raw', 'all', 'RMS']
dataset_list = [1, 2, 3]
rawBool_list = [True, False]
data_type_list = ['glove', 'restimulus']


if test:
    for data_type in data_type_list:
        for user in user_list:
            for dataset in dataset_list:

                if data_type == 'emg':
                    for emg_type in emg_types:
                        emg_data = auxf.getProcessedEMG(user = user, dataset=dataset, type_data=emg_type)
                        auxf.runCebraCheck(emg_data)

                else:
                    for rawBool in rawBool_list:
                        data = auxf.getProcessedData(user=user, dataset=dataset, mode=data_type, rawBool=rawBool)
                        auxf.runCebraCheck(data)










# self-supervised
                        
def trainSelfSupervised(emg_type: str, user: int):

    directory = f"./final_working/self_supervised/{emg_type}"
    
    batch_size = 256
    min_temp = 0.3


    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"

                       
    directory = f"./final_working/model_training/self_supervised/{emg_type}"
    auxf.ensure_directory_exists(directory=directory)


    emg_tr1 = auxf.getProcessedEMG(user = user, dataset=1, type_data=emg_type)
    emg_tr2 = auxf.getProcessedEMG(user = user, dataset=2, type_data=emg_type)
    emg_test = auxf.getProcessedEMG(user = user, dataset=3, type_data=emg_type)

    if emg_type == 'raw':
        rawBool = True
    else:
        rawBool = False

    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)



    cebra_model_def = CEBRA(
                model_architecture = 'offset10-model',
                batch_size= 256,
                temperature_mode='auto',
                learning_rate = 0.0001,
                max_iterations = iterations,
                #min_temperature=1.2,
                time_offsets = 25,
                output_dimension = 3, 
                device = "cuda_if_available",
                verbose = True,
                conditional='time', # because self-supervised
                distance = 'cosine' 
            )   


    model = deepcopy(cebra_model_def)

    model.partial_fit(emg_tr1)
    model.partial_fit(emg_tr2)

    loss_directory = f"{directory}/loss_plots"
    auxf.ensure_directory_exists(loss_directory)

    cebra.plot_loss(model)
    plt.savefig(f"{loss_directory}/{model_ID}_loss.png")

    temp_directory = f"{directory}/temp_lots"
    auxf.ensure_directory_exists(temp_directory)

    cebra.plot_temperature(model)
    plt.savefig(f"{temp_directory}/{model_ID}_temp.png")


    embed_directory = f"{directory}/embeddings"
    auxf.ensure_directory_exists(embed_directory)


    gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # for label as gesture

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=restim_test, required_stimulus=gesture)

        embedding = model.transform(emg_test[start:end])

        labels = restim_test[start: end]
        labels = labels.flatten()

        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        unique_labels = np.unique(all_labels)

        # Create a new colormap from jet, but make the lowest value white
        jet = plt.cm.get_cmap('jet', len(all_labels))
        colors = jet(np.linspace(0, 1, len(all_labels)))
        colors[0] = (1, 1, 1, 1)  # RGBA for white

        custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=len(colors))

        # Create a mapping from labels to colors
        color_map = dict(zip(unique_labels, custom_cmap(np.linspace(0, 1, len(unique_labels)))))

        # Assign colors to each data point
        point_colors = [color_map[label] for label in labels]



        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update({"color": gridline_color})


        # Customizing each subplot
        ax.set_facecolor('black')
        axis_pane_color = 'black'
        ax.xaxis.set_pane_color(axis_pane_color)
        ax.yaxis.set_pane_color(axis_pane_color)
        ax.zaxis.set_pane_color(axis_pane_color)

        ax.set_box_aspect(aspect = None, zoom = 0.85)




        legend_handles = [Patch(color=color_map[label], label=label) for label in unique_labels]




        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 7, alpha = 1)
        ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)
        ax.tick_params(axis='x', colors='white', size = 1)
        ax.xaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='y', colors='white', size = 1)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.tick_params(axis='z', colors='white', size = 1)
        ax.zaxis.set_tick_params(labelsize=labelsize) 

        fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 0.75), title = 'Gesture', fontsize = 10) 

        fig.suptitle(f"Self-Supervised, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
        
    
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{embed_directory}/{model_ID}_embedding_LABELS.png")
    

    # for label as time

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=restim_test, required_stimulus=gesture)

        embedding = model.transform(emg_test[start:end])

        ax.set_facecolor('black')

        cebra.plot_embedding(embedding, 
                             embedding_labels='time',
                            cmap = 'magma', 
                             markersize=7, 
                             ax = ax, 
                             alpha = 0.99)
        
        ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize)


        ax.tick_params(axis='x', colors='white', size = 1)
        ax.xaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='y', colors='white', size = 1)
        ax.yaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='z', colors='white', size = 1)
        ax.zaxis.set_tick_params(labelsize=labelsize) 

        ax.grid(True)



        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update({"color": gridline_color})


        # Customizing each subplot
        ax.set_facecolor('black')
        axis_pane_color = 'black'
        ax.xaxis.set_pane_color(axis_pane_color)
        ax.yaxis.set_pane_color(axis_pane_color)
        ax.zaxis.set_pane_color(axis_pane_color)

        ax.set_box_aspect(aspect = None, zoom = 0.85)

    
        fig.suptitle(f"Self-Supervised, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
        plt.tight_layout()
        plt.savefig(f"{embed_directory}/{model_ID}_embedding_CEBRA.png")

    




    model_directory = f"{directory}/models"
    auxf.ensure_directory_exists(model_directory)

    model.save(f"{model_directory}/{model_ID}.pt")


suptitlesize = 17
labelsize = 11
miniplotsize = 15

iterations = 1
gridline_color = 'dimgray'



for user in user_list:
    trainSelfSupervised(user = user, emg_type='all')
    trainSelfSupervised(user = user, emg_type='raw')
    
# trainSelfSupervised(user = 1, emg_type='RMS')
# trainSelfSupervised(user = 7, emg_type='RMS')
# trainSelfSupervised(user = 6, emg_type='RMS')


# TO DO : be able to load the model and perform same function instead of training the models. 
# TO DO; the xticks do not look good. 
