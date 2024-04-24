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
from sklearn.model_selection import train_test_split
import aux_functions as auxf
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch import nn
import cebra.models
import cebra.data
from scipy.signal import butter, lfilter
import os
import pandas as pd
from torch import nn
import seaborn as sns
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator



from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from skorch import NeuralNetRegressor

from joblib import dump
from joblib import load 

from sklearn.linear_model import LinearRegression


"""
This script is used to calculate the similarities between embeddings across users when the data is aligned 
in behaviour. This idea is from the CEBRA paper (Schneider, S., Lee, J. H., & Mathis, M. W. (2023). Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617(7960), 360-368.)

In practice, in this dataset, it is very difficult to find points in the dataset where the behaviour is aligned across all users, as there is human variability in how the task was performed. 
By visual inspection, small sections of data were found where the behaviour was approximately aligned, and correlation heatmaps show R2 values after fitting a linear model between behaviour-aligned
embeddings of pairs of users - one as the target and the other as the source. 

This was done for the two best performing regression models - CEBRA-Behaviour and UMAP. 

The results were not presented in the final results as, in practice, it is very difficult to reliably align the behaviour (hand joint states) to draw conclusive results. 

"""


params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)



heatmap_path = './embedding_consistency'

def getCorrelationHeatmap(user_list, model_type: str, gesture: int):

    matrix = np.zeros((len(user_list), len(user_list)))

    for index_i, i in enumerate(user_list):
        for index_j, j in enumerate(user_list):

            print(f'i: {i}, j: {j}')

            # gesture = 2

            user1 = user_list[index_i]            
            user2 = user_list[index_j]



            
            user = user_list[index_i]
            emg_user1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
            stim_user_1= auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
            glove_user_1 = (auxf.getMappedGlove(user = user, dataset=1)).T


            start1, end1 = auxf.cutStimTransition(stimulus_data=stim_user_1, required_stimulus=gesture)

            if model_type == 'UMAP':
                model_user1 = UMAP(n_components=3, n_neighbors=40, min_dist=0.0001, metric='euclidean')
                embedding_user1 = model_user1.fit_transform(emg_user1[start1:start1+350])
            
            elif model_type == 'TSNE':
                model = TSNE(n_components=3)
                embedding_user1 = model.fit_transform(emg_user1[start1:start1+350])
            
            elif model_type == 'CEBRA':

                cebra_model_user1 = CEBRA(
                    model_architecture = "offset10-model",
                    batch_size = 256,
                    temperature_mode="auto",
                    learning_rate = 0.0001,
                    max_iterations = 1,
                    time_offsets = 25,
                    output_dimension = 3,
                    device = "cuda_if_available",
                    verbose = True,
                    conditional='time_delta',
                    min_temperature=0.3,
                )
                
                cebra_model_user1.fit(emg_user1, glove_user_1)

                if gesture == 2:
                    embedding_user1 = cebra_model_user1.transform(emg_user1[start1:start1+350])
                
                if gesture == 7:
                    embedding_user1 = cebra_model_user1.transform(emg_user1[end1-350:end1])






            user = user_list[index_j]
            emg_user2 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
            stim_user_2= auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
            glove_user_2 = (auxf.getMappedGlove(user = user, dataset=1)).T

            start2, end2 = auxf.cutStimTransition(stimulus_data=stim_user_2, required_stimulus=gesture)


            model_user2 = UMAP(n_components=3, n_neighbors=40, min_dist=0.0001, metric='euclidean')
            embedding_user2 = model_user2.fit_transform(emg_user2[start2:start2+350])

            if model_type == 'UMAP':
                model_user2 = UMAP(n_components=3, n_neighbors=40, min_dist=0.0001, metric='euclidean')
                embedding_user2 = model_user2.fit_transform(emg_user2[start2:start2+350])
            
            elif model_type == 'TSNE':
                model_user2 = TSNE(n_components=3)
                embedding_user2 = model_user2.fit_transform(emg_user2[start2:start2+350])
            
            elif model_type == 'CEBRA':

                cebra_model_user2 = CEBRA(
                    model_architecture = "offset10-model",
                    batch_size = 256,
                    temperature_mode="auto",
                    learning_rate = 0.0001,
                    max_iterations = 1,
                    time_offsets = 25,
                    output_dimension = 3,
                    device = "cuda_if_available",
                    verbose = True,
                    conditional='time_delta',
                    min_temperature=0.3,
                )
                
                cebra_model_user2.fit(emg_user2, glove_user_2)

                if gesture == 2: 
                    embedding_user2 = cebra_model_user2.transform(emg_user2[start2:start2+350])

                if gesture == 7:
                    embedding_user2 = cebra_model_user2.transform(emg_user2[end2-350:end2])



                
            X = embedding_user1
            y = embedding_user2

            plotEmbedding(embedding_user1, embedding_user2, model_type, gesture, user1, user2)

            # fig, ax = plt.subplots()

            # # plt.plot(stim_user_1[start1:start1+350], color = 'blue', label = 'user1')
            # # # plt.show()
            # # # plt.show()
            # # plt.plot(stim_user_2[start2:start2+350], color = 'red', label = 'user2')
            # # plt.legend()


            # fig, ax = plt.subplots()
            # plt.plot(glove_user_1[start1:start1+350, 0], color = 'blue', label = f'user{user1}')
            # plt.plot(glove_user_2[start2:start1+350, 0], color = 'red', label = f'user{user2}')
            # plt.legend()
            # plt.show()

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)    

            matrix[index_i, index_j] = r2



    user_name_list = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7', 'AB8', 'AB9', 'AB10', 'Amp1', 'Amp2']



    plt.figure(figsize=(12, 8))  # Adjust the size as needed

    # mask = np.zeros_like(matrix, dtype=bool)
    # np.fill_diagonal(mask, True)

    mask = np.zeros_like(matrix, dtype = bool)
    mask[np.eye(mask.shape[0], dtype = bool)] = True

    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis', mask = mask, vmin = 0, vmax = 1)



    colorbar = ax.collections[0].colorbar
    colorbar.set_label(r'$R^2$')

    tick_positions = np.arange(0.5, len(user_name_list), 1)  # Centers of the squares
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(user_name_list, fontsize = 12)
    ax.set_yticklabels(user_name_list, rotation = 45, fontsize = 12)


    # Optional: Customize further (e.g., adding a title, axis labels, etc.)
    ax.set_title(f'{model_type}')
    # ax.set_xticklabels(user_list)  # Set the x-axis labels to user_list
    # ax.set_yticklabels(user_list)  # Set the y-axis labels to user_list

    # plt.xlabel('User')
    # plt.ylabel('User')

    # plt.tight_layout()
    plt.savefig(f"{heatmap_path}/test_heatmap_{model_type}.png")
    # Show the plot
    plt.show()

    return matrix






def plotEmbedding(embedding_u1, embedding_u2, dim_red_type: str, gesture: int, user1, user2):


    embeddings = [embedding_u1, embedding_u2]
    users = [user1, user2]

    x_limits = [min(embed[:, 0].min() for embed in embeddings), max(embed[:, 0].max() for embed in embeddings)]
    y_limits = [min(embed[:, 1].min() for embed in embeddings), max(embed[:, 1].max() for embed in embeddings)]
    z_limits = [min(embed[:, 2].min() for embed in embeddings), max(embed[:, 2].max() for embed in embeddings)]



    fig, axes = plt.subplots(1, 2, figsize=(6, 3), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()


    for i, ax in enumerate(axes):

        ax.set_facecolor('black')

        if dim_red_type != 'CEBRA':
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)

        else: 
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        labels = np.arange(0, len(embedding_u1), 1)
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


        ax.scatter(embeddings[i][:, 0], embeddings[0][:, 1], embeddings[i][:, 2], c = point_colors, s = 8, alpha = 1, edgecolor = 'white', linewidth = 0.25)


        ax.set_title(f"{dim_red_type}: User{users[i]}, gesture {gesture}" , color = 'white', fontsize = miniplotsize, y = 0.97)



        ax.tick_params(axis='x', colors='white', size = 1)
        ax.xaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='y', colors='white', size = 1)
        ax.yaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='z', colors='white', size = 1)
        ax.zaxis.set_tick_params(labelsize=labelsize) 


        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(4))

        ax.grid(True)



        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update({"color": gridline_color})


        # Customizing each subplot
        ax.set_facecolor('black')
        axis_pane_color = 'black'
        ax.xaxis.set_pane_color(axis_pane_color)
        ax.yaxis.set_pane_color(axis_pane_color)
        ax.zaxis.set_pane_color(axis_pane_color)

        ax.set_box_aspect(aspect = None, zoom = 0.75)

    
        #fig.suptitle(f"{type_training}, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)

    savefig_path = f"{embed_directory}/{dim_red_type}"
    auxf.ensure_directory_exists(savefig_path)
    plt.tight_layout()
    plt.close()
    plt.savefig(f"{savefig_path}/user{user1}_user{user2}_gesture{gesture}_embedding_{dim_red_type}_gesture{gesture}.png")






user_list__ = np.arange(1, 13)

embed_directory = './embedding_heatmaps/'

# --------------------------------------# 
suptitlesize = 10
labelsize = 9
miniplotsize = 12

iterations = 20000
gridline_color = 'dimgray'
# --------------------------------------# 


matrix_CEBRA7 = getCorrelationHeatmap(user_list=user_list__, model_type="CEBRA", gesture = 7)
np.save(f'{heatmap_path}/matrix_CEBRA7.npy', matrix_CEBRA7)


matrix_CEBRA2 = getCorrelationHeatmap(user_list=user_list__, model_type="CEBRA", gesture = 2)
np.save(f'{heatmap_path}/matrix_CEBRA2.npy', matrix_CEBRA2)


matrix_UMAP_2 = getCorrelationHeatmap(user_list=user_list__, model_type="UMAP", gesture=2)
np.save(f'{heatmap_path}/matrix_UMAP_2.npy', matrix_UMAP_2)


matrix_UMAP_7 = getCorrelationHeatmap(user_list=user_list__, model_type="UMAP", gesture=7)
np.save(f'{heatmap_path}/matrix_UMAP_7.npy', matrix_UMAP_7)




