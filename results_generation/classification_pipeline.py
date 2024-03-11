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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA




list_models = []
dim = 3 
k = 201 # to review

directory_loadmodels = "./results_generation/model_training/restimulus_classification"

for dirpath, dirnames, filenames in os.walk(directory_loadmodels):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        
        if file_path.__contains__(".pt"):

            model_path = file_path
            list_models.append(model_path)
            print(model_path)


def plot_lda_cebra(lda_embedding, cebra_embedding, restim_test, user, modelID):

    gestures = [1, 2, 3, 4, 5, 6 ,7, 8 ,9]

    for gesture in gestures:

        print('gesture', gesture)

        start, end = auxf.cutStimulus(restim_test, gesture)
        print("start", start)
        print("end", end)

        if gesture == 9:
            end = len(restim_test)

        restim_test_ = restim_test[start:end]

        labels = restim_test_
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

        lda_embedding_plt = lda_embedding[start:end, :]
        cebra_embedding_plt = cebra_embedding[start:end, :]


        fig, axes = plt.subplots(1, 2, figsize = (15 ,15), subplot_kw={'projection' : '3d'})

        miniplotsize = 14
        labelsize = 10

        axes[0].scatter(lda_embedding_plt[:, 0], lda_embedding_plt[:, 1], lda_embedding_plt[:, 2], label = "LDA Embedding", c = point_colors)
        axes[1].scatter(cebra_embedding_plt[:, 0], cebra_embedding_plt[:, 1], cebra_embedding_plt[:, 2], label = "CEBRA Embedding", c = point_colors)
        # axes[0].set_title(f"LDA Embedding, Gesture {gesture} and 'Rest', User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)
        # axes[1].set_title(f"CEBRA Embedding, Gesture {gesture} and 'Rest', User {user}" , color = 'white', fontsize = miniplotsize, y = 0.97)
        axes[0].set_title(f"LDA Embedding, Gesture {gesture} and 'Rest', User {user}" , color = 'white', fontsize = miniplotsize)
        axes[1].set_title(f"CEBRA Embedding, Gesture {gesture} and 'Rest', User {user}" , color = 'white', fontsize = miniplotsize)


        axes = axes.flatten()

        for i, ax in enumerate(axes):

            gridline_color = 'dimgray'

            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo["grid"].update({"color": gridline_color})


            # Customizing each subplot
            ax.set_facecolor('black')
            axis_pane_color = 'black'
            ax.xaxis.set_pane_color(axis_pane_color)
            ax.yaxis.set_pane_color(axis_pane_color)
            ax.zaxis.set_pane_color(axis_pane_color)
            ax.tick_params(axis='x', colors='white', size = 1)
            ax.xaxis.set_tick_params(labelsize=labelsize) 
            ax.tick_params(axis='y', colors='white', size = 1)
            ax.yaxis.set_tick_params(labelsize=labelsize)
            ax.tick_params(axis='z', colors='white', size = 1)
            ax.zaxis.set_tick_params(labelsize=labelsize) 
    


        legend_handles = [Patch(color=color_map[label], label=label) for label in unique_labels]
        fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 0.75), title = 'Gesture', fontsize = 10) 

        directory_results_plots = f'./results_generation/restimulus_classification/results_plots'
        auxf.ensure_directory_exists(directory_results_plots)

        plt.suptitle(f"{modelID}_user{user}_{gesture}")

        plt.savefig(f"{directory_results_plots}/LDA_CEBRA_{modelID}_{gesture}.png")





def runKNNComparison(model_path):

    model = cebra.CEBRA.load(model_path)

    directory = os.path.dirname(model_path)
    directory = os.path.dirname(directory)



    type_training, user, emg_type, batch_size, min_temp, iterations = auxf.extract_model_params(model_path)

    if emg_type == "raw":
        rawBool = True

    else: 
        rawBool = False

    model_ID = f"user_{user}_{emg_type}_batch_{batch_size}_mintemp{min_temp}_it{iterations}"



    emg_tr1 = auxf.getProcessedEMG(user = user, dataset=1, type_data=emg_type)
    emg_tr2 = auxf.getProcessedEMG(user = user, dataset=2, type_data=emg_type)
    emg_test = auxf.getProcessedEMG(user = user, dataset=3, type_data=emg_type)


    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)

    emg_tr_concat = np.concatenate([emg_tr1, emg_tr2])
    restim_tr_concat = np.concatenate([restim_tr1, restim_tr2])


    training_embedding = model.transform(emg_tr_concat)

    # this is the embedding that then I need to plot
    test_embedding = model.transform(emg_test)


    # cebra KNN
    knn_cebra = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')

    knn_cebra.fit(training_embedding, restim_tr_concat)

    knn_cebra_pred = knn_cebra.predict(test_embedding)

    cebra_knn_accuracy = accuracy_score(restim_test, knn_cebra_pred)


    # LDA 
    lda = LDA(n_components= dim)

    emg_lda_train = lda.fit_transform(emg_tr_concat, restim_tr_concat)

    # this also needs to be plotted
    emg_lda_test = lda.transform(emg_test)

    # #plot_lda_cebra(lda_embedding=emg_lda_test, 
    #                cebra_embedding=test_embedding, 
    #                restim_test=restim_test, 
    #                user=user, 
    #                modelID=model_ID)


    # LDA KNN
    knn_lda = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')

    knn_lda.fit(emg_lda_train, restim_tr_concat)

    knn_lda_pred = knn_lda.predict(emg_lda_test)

    lda_knn_accuracy = accuracy_score(restim_test, knn_lda_pred)


    # find difference between cebra and LDA accuracy (if > 0 : cebra outperformed)
    difference = cebra_knn_accuracy - lda_knn_accuracy

    if difference > 0:
        cebra_outpeform = True

    else:
        cebra_outpeform = False


    df_row = {'model_name' : model_ID,
              "emg_type" : emg_type,
              "min_temp" : min_temp,
    'dim' : dim, 
    'batch_size' : batch_size, 
    'user' : user, 
    'iterations' : iterations,
    "lda_knn_accuracy" : lda_knn_accuracy, 
    "cebra_knn_accuracy" : cebra_knn_accuracy,
    "% _ difference" : difference*100,
    "cebra_outperform" : cebra_outpeform}

    #df_row = [model_name, time_offset, dim, batch_size, user_tr, user_test, iterations, lda_knn_accuracy, cebra_knn_accuracy, difference*100, cebra_outpeform]

    # df_row = pd.DataFrame(data=[df_row])  # or any other index you prefer

    # results_stored = pd.read_csv("./classification_gridsearch/LDA_test/results/results_df_2_3dims.csv")

    # results_df = pd.concat([results_stored, df_row])
    # results_df.to_csv("./classification_gridsearch/LDA_test/results/results_df_2_3dims.csv")


    print("LDA ACC: ", lda_knn_accuracy)
    print("CEBRA ACC", cebra_knn_accuracy)

    print(df_row)

    return df_row


results_list = []

results_df_inter = pd.DataFrame(columns = ['model_name',
              'emg_type',
              'min_temp',
              'dim', 
              'batch_size', 
              'user',
              'iterations',
              "lda_knn_accuracy", 
              "cebra_knn_accuracy",
              "% _ difference",
              "cebra_outperform"
              ])



directory_results_df = f'./results_generation/restimulus_classification/results_dataframes'
auxf.ensure_directory_exists(directory_results_df)


results_path = f"{directory_results_df}/LDA_Comparison.csv"
results_df_inter.to_csv(results_path)


def runComparison():
    for model in list_models:
        df_row = runKNNComparison(model)
        results_list.append(df_row)
        df_row = pd.DataFrame(data=[df_row]) 

        results_stored = pd.read_csv(results_path)
        results_df = pd.concat([results_stored, df_row])
        results_df.to_csv(results_path, index = False)



results_df = pd.read_csv("results_generation/restimulus_classification/results_dataframes/LDA_Comparison_saved.csv")





mintemp_list = [0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5]
mean_mintemp_lists = []

for min_temp in mintemp_list: 

    mintemp_list = []
    mintemp_list.append(min_temp)
    mean_improvement = round(results_df["% _ difference"][results_df['min_temp'] == min_temp].mean(), 3)
    mean_accuracy = round(results_df["cebra_knn_accuracy"][results_df['min_temp'] == min_temp].mean(), 3)

    print(mean_accuracy)
    mintemp_list.append(mean_improvement)
    mintemp_list.append(mean_accuracy)

    mean_mintemp_lists.append(mintemp_list)


mintemp_results_path = f"{directory_results_df}/LDA_Comparison_MinTemp.csv"
    
mintemp_df = pd.DataFrame(mean_mintemp_lists, columns = ['min_temp', 'mean_improvement', 'mean_accuracy']) 
mintemp_df.to_csv(mintemp_results_path)





