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
emg_types = ['raw', 'all'] # remove 'RMS'
dataset_list = [1, 2, 3]
rawBool_list = [True, False]
data_type_list = ['glove', 'restimulus']



# this is a check to ensure that there will be no running error with the datasets 

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
                        
def trainSelfSupervised(emg_type: str, user: int, min_temp):

    
    batch_size = 256

    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"

                       
    directory = f"./results_generation/model_training/self_supervised/{emg_type}"
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
                min_temperature= min_temp,
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


    model_directory = f"{directory}/models"
    auxf.ensure_directory_exists(model_directory)

    model_path = f"{model_directory}/{model_ID}.pt"

    model.save(model_path)

    return model_path

    


    # gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # # for label as gesture

    # fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    # axes = axes.flatten()

    # for i, ax in enumerate(axes):

    #     gesture = gestures[i]

    #     start, end = auxf.cutStimTransition(stimulus_data=restim_test, required_stimulus=gesture)

    #     embedding = model.transform(emg_test[start:end])

    #     labels = restim_test[start: end]
    #     labels = labels.flatten()

    #     all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     unique_labels = np.unique(all_labels)

    #     # Create a new colormap from jet, but make the lowest value white
    #     jet = plt.cm.get_cmap('jet', len(all_labels))
    #     colors = jet(np.linspace(0, 1, len(all_labels)))
    #     colors[0] = (1, 1, 1, 1)  # RGBA for white

    #     custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=len(colors))

    #     # Create a mapping from labels to colors
    #     color_map = dict(zip(unique_labels, custom_cmap(np.linspace(0, 1, len(unique_labels)))))

    #     # Assign colors to each data point
    #     point_colors = [color_map[label] for label in labels]



    #     for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #         axis._axinfo["grid"].update({"color": gridline_color})


    #     # Customizing each subplot
    #     ax.set_facecolor('black')
    #     axis_pane_color = 'black'
    #     ax.xaxis.set_pane_color(axis_pane_color)
    #     ax.yaxis.set_pane_color(axis_pane_color)
    #     ax.zaxis.set_pane_color(axis_pane_color)

    #     ax.set_box_aspect(aspect = None, zoom = 0.85)




    #     legend_handles = [Patch(color=color_map[label], label=label) for label in unique_labels]




    #     ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 7, alpha = 1)
    #     ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)
    #     ax.tick_params(axis='x', colors='white', size = 1)
    #     ax.xaxis.set_tick_params(labelsize=labelsize) 
    #     ax.tick_params(axis='y', colors='white', size = 1)
    #     ax.yaxis.set_tick_params(labelsize=labelsize)
    #     ax.tick_params(axis='z', colors='white', size = 1)
    #     ax.zaxis.set_tick_params(labelsize=labelsize) 

    #     fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 0.75), title = 'Gesture', fontsize = 10) 

    #     fig.suptitle(f"Self-Supervised, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
        
    
    # plt.tight_layout()
    # plt.grid(False)
    # plt.savefig(f"{embed_directory}/{model_ID}_embedding_LABELS.png")
    

    # # for label as time

    # fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    # axes = axes.flatten()

    # for i, ax in enumerate(axes):

    #     gesture = gestures[i]

    #     start, end = auxf.cutStimTransition(stimulus_data=restim_test, required_stimulus=gesture)

    #     embedding = model.transform(emg_test[start:end])

    #     ax.set_facecolor('black')

    #     # cebra.plot_embedding(embedding, 
    #     #                      embedding_labels='time',
    #     #                     cmap = 'magma', 
    #     #                      markersize=7, 
    #     #                      ax = ax, 
    #     #                      alpha = 0.99, 
    #     #                      edgecolors = 'white', 
    #     #                      linewidths = 0.15)

    #     labels = np.arange(0, len(restim_test[start: end]), 1)
    #     print(labels)
    #     labels = labels.flatten()

    #     all_labels = labels

    #     unique_labels = np.unique(all_labels)

    #     # Create a new colormap from jet, but make the lowest value white
    #     jet = plt.cm.get_cmap('plasma', len(all_labels))
    #     colors = jet(np.linspace(0, 1, len(all_labels)))
    #     colors[0] = (1, 1, 1, 1)  # RGBA for white

    #     custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=len(colors))

    #     # Create a mapping from labels to colors
    #     color_map = dict(zip(unique_labels, custom_cmap(np.linspace(0, 1, len(unique_labels)))))

    #     # Assign colors to each data point
    #     point_colors = [color_map[label] for label in labels]



    #     # cebra.plot_embedding(embedding, 
    #     #                      embedding_labels='time',
    #     #                     cmap = 'cebra', 
    #     #                      markersize=7, 
    #     #                      ax = ax, 
    #     #                      alpha = 0.99, 
    #     #                      edgecolors = 'white', 
    #     #                      linewidths = 0.15)


    #     ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 10, alpha = 1, edgecolor = 'white', linewidth = 0.25)

        
    #     ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize)


    #     ax.tick_params(axis='x', colors='white', size = 1)
    #     ax.xaxis.set_tick_params(labelsize=labelsize) 
    #     ax.tick_params(axis='y', colors='white', size = 1)
    #     ax.yaxis.set_tick_params(labelsize=labelsize) 
    #     ax.tick_params(axis='z', colors='white', size = 1)
    #     ax.zaxis.set_tick_params(labelsize=labelsize) 

    #     ax.grid(True)



    #     for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #         axis._axinfo["grid"].update({"color": gridline_color})


    #     # Customizing each subplot
    #     ax.set_facecolor('black')
    #     axis_pane_color = 'black'
    #     ax.xaxis.set_pane_color(axis_pane_color)
    #     ax.yaxis.set_pane_color(axis_pane_color)
    #     ax.zaxis.set_pane_color(axis_pane_color)

    #     ax.set_box_aspect(aspect = None, zoom = 0.85)

    
    #     fig.suptitle(f"Self-Supervised, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
    #     plt.tight_layout()
    #     plt.savefig(f"{embed_directory}/{model_ID}_embedding_CEBRA.png")

    





# ---------------------------------------# 
    
def trainBehaviourContrastive(emg_type: str, user: int, min_temp):

    directory = f"./results_generation/model_training/behaviour_contrastive/{emg_type}"
    
    #min_temp = 0.5

    if emg_type == 'raw':
        rawBool = True
        batch_size = 32
    else:
        rawBool = False
        batch_size = 256




    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"



                       
    auxf.ensure_directory_exists(directory=directory)


    emg_tr1 = auxf.getProcessedEMG(user = user, dataset=1, type_data=emg_type)
    emg_tr2 = auxf.getProcessedEMG(user = user, dataset=2, type_data=emg_type)
    # emg_test = auxf.getProcessedEMG(user = user, dataset=3, type_data=emg_type)

    glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='glove', rawBool = rawBool)
    glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='glove', rawBool = rawBool)
    glove_test = auxf.getProcessedData(user = user, dataset = 3, mode='glove', rawBool = rawBool)



    # restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    # restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    # restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)




    cebra_model_def = CEBRA(
                model_architecture = 'offset10-model',
                batch_size= 256,
                temperature_mode='auto',
                learning_rate = 0.0001,
                max_iterations = iterations,
                min_temperature= min_temp, 
                time_offsets = 25,
                output_dimension = 3, 
                device = "cuda_if_available",
                verbose = True,
                conditional='time_delta',
                distance = 'cosine' 
            )   


    model = deepcopy(cebra_model_def)

    model.partial_fit(emg_tr1, glove_tr1)
    model.partial_fit(emg_tr2, glove_tr2)

    loss_directory = f"{directory}/loss_plots"
    auxf.ensure_directory_exists(loss_directory)

    cebra.plot_loss(model)
    plt.savefig(f"{loss_directory}/{model_ID}_loss.png")

    temp_directory = f"{directory}/temp_lots"
    auxf.ensure_directory_exists(temp_directory)

    cebra.plot_temperature(model)
    plt.savefig(f"{temp_directory}/{model_ID}_temp.png")


    model_directory = f"{directory}/models"
    auxf.ensure_directory_exists(model_directory)

    model_path = f"{model_directory}/{model_ID}.pt"

    model.save(model_path)

    return model_path


def trainRestimulus(emg_type: str, user: int, min_temp):

    directory = f"./results_generation/model_training/restimulus_classification/{emg_type}"
    


    if emg_type == 'raw':
        rawBool = True
        batch_size = 32

    else:
        rawBool = False
        batch_size = 256


    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"

                       
    auxf.ensure_directory_exists(directory=directory)


    emg_tr1 = auxf.getProcessedEMG(user = user, dataset=1, type_data=emg_type)
    emg_tr2 = auxf.getProcessedEMG(user = user, dataset=2, type_data=emg_type)
    # emg_test = auxf.getProcessedEMG(user = user, dataset=3, type_data=emg_type)




    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)



    cebra_model_def = CEBRA(
                model_architecture = 'offset10-model',
                batch_size= 256, # had to lower bc memory
                temperature_mode='auto',
                learning_rate = 0.0001,
                max_iterations = iterations,
                min_temperature= min_temp, 
                time_offsets = 25,
                output_dimension = 3, 
                device = "cuda_if_available",
                verbose = True,
                conditional='time_delta',
                distance = 'cosine' 
            )   


    model = deepcopy(cebra_model_def)

    model.partial_fit(emg_tr1, restim_tr1)
    model.partial_fit(emg_tr2, restim_tr2)

    loss_directory = f"{directory}/loss_plots"
    auxf.ensure_directory_exists(loss_directory)

    cebra.plot_loss(model)
    plt.savefig(f"{loss_directory}/{model_ID}_loss.png")

    temp_directory = f"{directory}/temp_lots"
    auxf.ensure_directory_exists(temp_directory)

    cebra.plot_temperature(model)
    plt.savefig(f"{temp_directory}/{model_ID}_temp.png")


    model_directory = f"{directory}/models"
    auxf.ensure_directory_exists(model_directory)

    model_path = f"{model_directory}/{model_ID}.pt"

    model.save(model_path)

    return model_path



def plotGestures(model_path: str):
    
    model = cebra.CEBRA.load(model_path)

    directory = os.path.dirname(model_path)
    directory = os.path.dirname(directory)

    
    print("directory", directory)

    type_training, user, emg_type, batch_size, min_temp, iterations = auxf.extract_model_params(model_path)

    if emg_type == "raw":
        rawBool = True
    
    else: 
        rawBool = False

    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"


    emg_test = auxf.getProcessedEMG(user = user, dataset=3, type_data=emg_type)
    glove_test = auxf.getProcessedData(user = user, dataset = 3, mode='glove', rawBool = rawBool)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)


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



        labels = np.arange(0, len(restim_test[start: end]), 1)
        labels = labels.flatten()

        all_labels = labels

        unique_labels = np.unique(all_labels)

        # Create a new colormap from jet, but make the lowest value white
        jet = plt.cm.get_cmap('plasma', len(all_labels))
        colors = jet(np.linspace(0, 1, len(all_labels)))
        colors[0] = (1, 1, 1, 1)  # RGBA for white

        custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=len(colors))

        # Create a mapping from labels to colors
        color_map = dict(zip(unique_labels, custom_cmap(np.linspace(0, 1, len(unique_labels)))))

        # Assign colors to each data point
        point_colors = [color_map[label] for label in labels]



        # cebra.plot_embedding(embedding, 
        #                      embedding_labels='time',
        #                     cmap = 'cebra', 
        #                      markersize=7, 
        #                      ax = ax, 
        #                      alpha = 0.99, 
        #                      edgecolors = 'white', 
        #                      linewidths = 0.15)


        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 10, alpha = 1, edgecolor = 'white', linewidth = 0.25)

        ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)


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
        print("saving fig in ",f"{embed_directory}/{model_ID}_embedding_CEBRA.png" )

    






# --------------------------------------# 
suptitlesize = 17
labelsize = 11
miniplotsize = 15

iterations = 10000
gridline_color = 'dimgray'


# plotGestures("results_generation/model_training/behaviour_contrastive/all/models/user_4_all_batch_256_mintemp0.2_it15000.pt")
# print('finished')


# directory = "./results_generation/model_training"

# for dirpath, dirnames, filenames in os.walk(directory):
#     for filename in filenames:
#         file_path = os.path.join(dirpath, filename)
        
#         if file_path.__contains__(".pt"):
#             print(file_path)
#             plotGestures(file_path)
#             print("done gestures")

# # for user in user_list:
# #     trainBehaviourContrastive(user = user, emg_type='all')
    


for user in user_list: 

    emg_type = 'all'
    model_path = trainRestimulus(emg_type=emg_type, user=user, min_temp=0.1)
    plotGestures(model_path)

    model_path = trainBehaviourContrastive(emg_type=emg_type, user=user, min_temp=0.1)
    plotGestures(model_path)



# trainSelfSupervised(user = 1, emg_type='RMS')
# trainSelfSupervised(user = 7, emg_type='RMS')
# trainSelfSupervised(user = 6, emg_type='RMS')


# TO DO : be able to load the model and perform same function instead of training the models. 
# TO DO; the xticks do not look good. 
# i think it might be better with no min offset
    
