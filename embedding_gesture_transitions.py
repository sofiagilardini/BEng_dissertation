import sys

sys.path.append("/home/sofia/BEng_dissertation")


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from umap import UMAP

from autoencoders_skorch import AutoEncoder, AutoEncoderNet, VariationalAutoEncoder, VariationalAutoEncoderNet

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from joblib import dump
from joblib import load 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# --------------------------------------# 
suptitlesize = 17
labelsize = 13
miniplotsize = 20

gridline_color = 'dimgray'
# --------------------------------------# 




def plotGestures_dimRed(user, dim_red_type: str, umap_neighbors: None):

    """
    Plotting the embeddings of gesture transitions for different dimred methods
    If UMAP is being used, you can specify the number of neighbours. 
    
    """

    if dim_red_type == 'UMAP':
        modelID = f"UMAP_neighbours{umap_neighbors}"
    
    else:
        modelID = f"{dim_red_type}"


    embed_directory = f'./embedding_visualisation/{modelID}'
    auxf.ensure_directory_exists(embed_directory)

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


    if dim_red_type == 'UMAP':
        model = UMAP(n_components=3, n_neighbors=umap_neighbors, min_dist=0.0001, metric='cosine')

    elif dim_red_type == 'AE':
            max_epochs = 600
            model = AutoEncoderNet(
                AutoEncoder,
                module__num_units=3,
                module__input_size=48,
                lr=0.0001,
                max_epochs=max_epochs,
            )
            modelID = f'AE_epochs{max_epochs}'

    elif dim_red_type == 'PCA':
        model = PCA(n_components=3)

    if dim_red_type != 'AE':
        model.fit(emg_trainval)

    elif dim_red_type == 'AE':
        model.fit(emg_trainval.astype(np.float32), emg_trainval.astype(np.float32))


    emg_test = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
    stim_test = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)
    glove_test = (auxf.getMappedGlove(user = user, dataset=3)).T

    gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # for label as gesture

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=stim_test, required_stimulus=gesture)

        if dim_red_type != 'AE':
            embedding = model.transform(emg_test[start:end])
        
        elif dim_red_type == 'AE':
            _, embedding = model.forward(emg_test.astype(np.float32)[start:end])


        labels = stim_test[start: end]
        labels = labels.flatten()

        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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


        if gesture <= 8:
            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)

        else: 
            ax.set_title(f"Gesture {gesture} to end, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)


        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)


        ax.tick_params(axis='x', colors='white', size = 1)
        ax.xaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='y', colors='white', size = 1)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.tick_params(axis='z', colors='white', size = 1)
        ax.zaxis.set_tick_params(labelsize=labelsize) 

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.zaxis.set_major_locator(MaxNLocator(4))

        plt.tight_layout()

        lgd = fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.975, 0.73), title = 'Gesture', fontsize = 13) 

        #fig.suptitle(f"{type_training}, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)

    
    modelID = f'user{user}'
    
    #plt.tight_layout()
    plt.grid(False)

    plt.savefig(f"{embed_directory}/{modelID}_embedding_LABELS.png", bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
    # plt.savefig(f"{embed_directory}/{model_ID}_embedding_LABELS.png")


    # for label as time

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=stim_test, required_stimulus=gesture)

        if dim_red_type != 'AE':
            embedding = model.transform(emg_test[start:end])
        
        elif dim_red_type == 'AE':
            _, embedding = model.forward(emg_test.astype(np.float32)[start:end])


        ax.set_facecolor('black')



        labels = np.arange(0, len(stim_test[start: end]), 1)
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



        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 10, alpha = 1, edgecolor = 'white', linewidth = 0.1)

        if gesture <= 8:
            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)

        else: 
            ax.set_title(f"Gesture {gesture} to end, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)



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

        ax.set_box_aspect(aspect = None, zoom = 0.85)

        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)

        model_ID = f'user{user}'
    


        #fig.suptitle(f"{type_training}, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
        plt.tight_layout()
        plt.savefig(f"{embed_directory}/{model_ID}_embedding_CEBRA.png")
        print("saving fig in ",f"{embed_directory}/{model_ID}_embedding_CEBRA.png" )





def plotGestures_loadedmodels(user: int, modelID_path: str):

    """

    For a CEBRA modelID (Behaviour/Hybrid/Time) plot the embeddings of the gesture transitions
    
    
    """


    embed_directory = f'./embedding_visualisation/{modelID_path}'
    auxf.ensure_directory_exists(embed_directory)

    model_dir = f'./saved_models/{modelID_path}'
    auxf.ensure_directory_exists(model_dir)


    # pipe = load(f"./glove_linear_mapping/reg_saved/saved_pipes/user{user}_pipe.joblib")
    # model = pipe[0]

    # model_ID = f"user{user}"

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


    model = CEBRA.load(f'./saved_models/{modelID_path}/user{user}_regression.pt')

    emg_test = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
    stim_test = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)
    glove_test = (auxf.getMappedGlove(user = user, dataset=3)).T

    gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # for label as gesture

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=stim_test, required_stimulus=gesture)

        embedding = model.transform(emg_test[start:end])

        labels = stim_test[start: end]
        labels = labels.flatten()

        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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


        if gesture <= 8:
            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)

        else: 
            ax.set_title(f"Gesture {gesture} to end, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)


        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)


        ax.tick_params(axis='x', colors='white', size = 1)
        ax.xaxis.set_tick_params(labelsize=labelsize) 
        ax.tick_params(axis='y', colors='white', size = 1)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.tick_params(axis='z', colors='white', size = 1)
        ax.zaxis.set_tick_params(labelsize=labelsize) 

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.zaxis.set_major_locator(MaxNLocator(4))

        plt.tight_layout()

        lgd = fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.975, 0.73), title = 'Gesture', fontsize = 13) 

        #fig.suptitle(f"{type_training}, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)

        
    
    #plt.tight_layout()
    plt.grid(False)

    plt.savefig(f"{embed_directory}/user{user}_embedding_LABELS.png", bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
    # plt.savefig(f"{embed_directory}/{model_ID}_embedding_LABELS.png")


    # for label as time

    fig, axes = plt.subplots(3, 3, figsize=(19, 18), subplot_kw={'projection': '3d'})  # All subplots are 3D

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        gesture = gestures[i]

        start, end = auxf.cutStimTransition(stimulus_data=stim_test, required_stimulus=gesture)

        embedding = model.transform(emg_test[start:end])

        ax.set_facecolor('black')



        labels = np.arange(0, len(stim_test[start: end]), 1)
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

        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = point_colors, s = 10, alpha = 1, edgecolor = 'white', linewidth = 0.1)

        if gesture <= 8:
            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)

        else: 
            ax.set_title(f"Gesture {gesture} to end, User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)



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

        ax.set_box_aspect(aspect = None, zoom = 0.85)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    
        #fig.suptitle(f"{type_training}, User: {user}, Model ID: {model_ID}", fontsize = suptitlesize)
        plt.tight_layout()
        plt.savefig(f"{embed_directory}/user{user}_embedding_CEBRA.png")




def StatsAndEstimates(regression_dir: str):

    """
    The purpose of this function is to generate the plots appertaining to each model. This includes statistics per user, 
    as well as the distribution of test performance across DoA. 

    It also plots the 'test' estimates that were produced in regression and plots the comparison 
    trajectories for the estimates and the measured signal. 
    
    """

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    user_list = np.arange(1, 13)

    
    for user in user_list: 

        r2_scores = pd.read_csv(f"{regression_dir}/test_r2_results.csv")
        r2_scores = r2_scores.drop('Unnamed: 0', axis =1)

        r2_row = r2_scores[r2_scores['user'] == user]

        estimate = pd.read_csv(f'{regression_dir}/estimates/test_user{user}.csv')
        estimate = auxf.lowpass_filter(estimate, 950, 2000)

        # the test dataset is dataset 3
        emg_test = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
        stim_test= auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)
        glove_test = (auxf.getMappedGlove(user = user, dataset=3)).T


        estimate_df = pd.DataFrame(estimate)
        estimate_df['Time'] = estimate_df.index
        melted_est = pd.melt(estimate_df, id_vars = ['Time'], value_vars = estimate_df.columns[:-1], var_name = 'Channel', value_name = 'Estimate')
        melted_est['Channel'] = melted_est['Channel'].astype(int)

        glove_df = pd.DataFrame(glove_test)
        glove_df['Time'] = glove_df.index
        melted_glove = pd.melt(glove_df, id_vars = ['Time'], value_vars = glove_df.columns[:-1], var_name = 'Channel', value_name = 'Measured')

        window_stride_ms = 52

        merged_df = pd.merge(melted_glove, melted_est, on=['Time', 'Channel'])
        merged_df['actual_time_ms'] = merged_df['Time'] * window_stride_ms
        merged_df['actual_time_s'] = merged_df['actual_time_ms'] / 1000.0


        # sns.set_style("darkgrid", {"axes.facecolor": ".9"})

        g = sns.FacetGrid(merged_df, col='Channel', col_wrap=1, sharey=True, height=2.5, aspect=3)
        g = g.map(plt.plot, 'actual_time_s', 'Measured', color='black', label='Measured')
        g = g.map(plt.plot, 'actual_time_s', 'Estimate', color='blue', label='Estimate')

        for ax in g.axes.flatten():
            ax.xaxis.grid(False)  # Disable x-axis gridlines

        time_range = merged_df['actual_time_s'].unique()
        min_time, max_time = time_range.min(), time_range.max()


        channel_names = ['Thumb rotation', 'Thumb flexion', 'Index flexion', 'Middle flexion', 'Ring/little flexion']  
        i = 0


        for ax, channel_name in zip(g.axes.flat, channel_names):

            r2_ch = round(r2_row[f'test_r2_ch{i}'].values[0], 2)


            ax.set_xlim(min_time, max_time)

            ax.set_xticks(np.arange(0, 300, 50))  

            ax.set_ylabel(channel_name)
            ax.set_title(f"$R^2$ = {r2_ch}", fontdict={'fontname': 'serif', 'fontsize': 17})

            ax.set_xlabel('Time [s]', fontdict={'fontname': 'serif', 'fontsize' : 15})
            ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'serif', 'fontsize' : 15})
            i+=1

            for label in ax.get_xticklabels():
                label.set_fontname('serif')
                label.set_fontsize(15)

            for label in ax.get_yticklabels():
                label.set_fontname('serif')
                label.set_fontsize(15)




        plt.subplots_adjust(hspace=0.25) 
        savefig_path = f"{regression_dir}/estimate_plots_test"
        auxf.ensure_directory_exists(savefig_path)
        plt.savefig(f"{savefig_path}/CEBRA_user{user}.png")
        plt.close()

    # -------- end of per-user ----------- #

    
    # this will be needed later for the box plot 
    r2_channel_mean_test = [round(r2_scores[f"test_r2_ch{channel}"].mean(), 2) for channel in range(5)]

    user_r2_mean_test = [round(r2_scores[r2_scores['user'] == user].iloc[:, 1:].mean(axis=0).mean(), 2) for user in user_list]

    cv_scores = pd.read_csv(f'{regression_dir}/bestmodel_results.csv')
    cv_scores = cv_scores.drop('Unnamed: 0', axis =1)

    r2_comp_df = cv_scores.copy()
    r2_comp_df = r2_comp_df.drop('best_params', axis = 1)
    r2_comp_df = r2_comp_df.rename(columns = {'best_score_r2': 'Cross-Validation'})
    r2_comp_df['Test'] = user_r2_mean_test

    r2_comp_melted = pd.melt(r2_comp_df, 
                            id_vars = 'user', 
                            value_vars = ['Cross-Validation', 'Test'],
                            var_name = 'score_type', 
                            value_name='R2 Score' )




    fig, ax = plt.subplots(figsize = (10, 6))
    palette_name = 'PuBuGn'

    user_list_names = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7', 'AB8', 'AB9', 'AB10', 'Amp1', 'Amp2']

    # r2_comp_melted['user'] = r2_comp_melted['user'].map(dict(enumerate(user_list_names)))

    user_dict = {i + 1: user_list_names[i] for i in range(len(user_list_names))}

    r2_comp_melted['user'] = r2_comp_melted['user'].map(user_dict)


    sns.barplot(x='user', 
                y='R2 Score', 
                hue='score_type', 
                data=r2_comp_melted, 
                palette= palette_name, 
                ax=ax)


    legend = ax.legend(loc = 'upper left')
    legend.set_title('')
    legend_texts = legend.get_texts() 
    legend.set_frame_on(False)

    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('$R^2$ Score')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('serif')
        label.set_fontsize(15)
    ax.set_xlabel(ax.get_xlabel(), fontdict={'fontname': 'serif', 'fontsize': 17})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'serif', 'fontsize': 17})

    for text in legend_texts:
        text.set_fontname('serif')
        text.set_fontsize(15)


    savefig_path = f"{regression_dir}/stats_plots"
    auxf.ensure_directory_exists(savefig_path)
    plt.savefig(f"{savefig_path}/r2_cv_test_allusers2.png")
    plt.close()



    fig, ax = plt.subplots(figsize = (10, 6))
    palette_name = 'crest'

    r2_scores_toplot = r2_scores.copy()

    ['Thumb rotation', 'Thumb flexion', 'Index flexion', 'Middle flexion', 'Ring/little flexion']  

    r2_scores_toplot = r2_scores_toplot.rename(columns = {'test_r2_ch0' : 'Thumb Rotation', 
                                                        'test_r2_ch1' : 'Thumb flexion', 
                                                        'test_r2_ch2' : 'Index flexion', 
                                                        'test_r2_ch3' : 'Middle flexion', 
                                                        'test_r2_ch4' : 'Ring/little flexion'})

    r2_by_channel_melted = pd.melt(r2_scores_toplot, 
                            id_vars = 'user', 
                            value_vars = ['Thumb Rotation', 'Thumb flexion', 'Index flexion', 'Middle flexion', 'Ring/little flexion'],
                            var_name = 'channel', 
                            value_name='R2 Score' )




    user_list_names = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7', 'AB8', 'AB9', 'AB10', 'Amp1', 'Amp2']

    # r2_comp_melted['user'] = r2_comp_melted['user'].map(dict(enumerate(user_list_names)))

    user_dict = {i + 1: user_list_names[i] for i in range(len(user_list_names))}

    r2_by_channel_melted['user'] = r2_by_channel_melted['user'].map(user_dict)


    sns.barplot(x='user', 
                y='R2 Score', 
                hue='channel', 
                data=r2_by_channel_melted, 
                palette=palette_name, 
                ax=ax)


    # legend = ax.legend(loc = 'upper left')
    legend = ax.legend(loc='upper center', ncol = 3)

    legend.set_title('')
    legend_texts = legend.get_texts() 
    legend.set_frame_on(False)

    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('$R^2$ Score')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('serif')
        label.set_fontsize(15)
    ax.set_xlabel(ax.get_xlabel(), fontdict={'fontname': 'serif', 'fontsize': 17})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'serif', 'fontsize': 17})

    for text in legend_texts:
        text.set_fontname('serif')
        text.set_fontsize(13)

    savefig_path = f"{regression_dir}/stats_plots"
    auxf.ensure_directory_exists(savefig_path)
    plt.savefig(f"{savefig_path}/r2_bychannel_allusers.png")
    plt.close()






# --------------------------------------------- #

user_list = np.arange(1, 13)

time_offset_list_36 = [10, 15, 20, 25]
mintemp_list = [0.3, 0.5, 0.8]



# --- CEBRA-Behaviour ---- # 


for user in user_list:
    for min_temp in mintemp_list:
        for time_offset in time_offset_list_36:

            model_offset = 36
            modelID = f"offset-{model_offset}_timeoff{time_offset}_mintemp{min_temp}"
            modelID_path = f'BehContr/{modelID}'
            print(f'on user {user} for modelid {modelID_path}')
            plotGestures_loadedmodels(user = user, modelID_path=modelID_path)


# ---- CEBRA-Hybrid ----- # 

for user in user_list:
    for min_temp in mintemp_list:
        for time_offset in time_offset_list_36:

            model_offset = 36
            modelID = f"offset-{model_offset}_timeoff{time_offset}_mintemp{min_temp}"
            modelID_path = f'Hybrid/{modelID}'
            print(f'on user {user} for modelid {modelID_path}')
            plotGestures_loadedmodels(user = user, modelID_path=modelID_path)




# --- PCA, AE, UMAP ---- #

for user in user_list:
    print("on user", user)
    plotGestures_dimRed(user = user, umap_neighbors=40, dim_red_type='UMAP')    
    plotGestures_dimRed(user = user, umap_neighbors=None, dim_red_type='PCA')    
    plotGestures_dimRed(user = user, umap_neighbors=None, dim_red_type='AE')    

