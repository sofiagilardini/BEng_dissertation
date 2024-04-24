
import sys

sys.path.append("./BEng_dissertation")

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
import pandas as pd
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from autoencoders_skorch import AutoEncoder_1, AutoEncoder_2, AutoEncoderNet, VariationalAutoEncoder, VariationalAutoEncoderNet

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from joblib import dump
from joblib import load 
import random


"""

This script performs a regression gridsearch for CEBRA models using a MLP. CSV files of the estimates are saved, as well as plots of the gesture transitions, comparing
the predicted hand DOA and the ground truth. 

01/04/24 integrated with regression_plot_gen.py

12/04/24 restructured to allow for loaded models rather than training in-situ


"""


params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)


user_list = np.arange(1, 13)
print(user_list)


def runMlpGridsearch_CEBRA_loaded_models(modelID: str):

    """
    The purpose of this function is to load a type of CEBRA model (BehContr/SelfSup/Hybrid) and perform regression 
    using the embeddings of the model by performing gridsearch and cross-validation over MLP architecures. 
    
    """
    
    modelID_path = modelID

    saved_model_path = "./saved_models"
        
    user_results = []
    best_model_list = []
    r2_scores = []

    for user in user_list:

        print("on user", user)

        emg1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
        emg2 = auxf.getProcessedEMG(user = user, dataset= 2, type_data = 'all')

        stim1 = auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
        stim2 = auxf.getProcessedData(user = user, dataset=2, mode = 'restimulus', rawBool=False)

        glove1 = (auxf.getMappedGlove(user = user, dataset=1)).T
        glove2 = (auxf.getMappedGlove(user = user, dataset=2)).T

        # create trainval
        emg_trainval = np.concatenate((emg1, emg2))
        glove_trainval = np.concatenate((glove1, glove2))

        emg3 = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
        glove3 = (auxf.getMappedGlove(user = user, dataset=3)).T
        stim3 = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)


        pipeline_mlp = Pipeline([
            ('mlp', MLPRegressor(max_iter=350))
            ])


        param_grid = {
            'mlp__hidden_layer_sizes' : [(256, 256, 256), (256, 256), (150, 150)]
        }


        loaded_cebra_path = f"{saved_model_path}/{modelID_path}/user{user}_regression.pt"
        cebra_model = cebra.CEBRA.load(loaded_cebra_path)


        emg_trainval_transformed = cebra_model.transform(emg_trainval)  
        emg_test_transformed = cebra_model.transform(emg3)  

        grid_search_mlp = GridSearchCV(pipeline_mlp, param_grid, cv=3, scoring='r2')
        grid_search_mlp.fit(emg_trainval_transformed, glove_trainval)  # glove_trainval are the labels


        best_params_cebra = grid_search_mlp.best_params_
        best_score_cebra = grid_search_mlp.best_score_


        user_row = [user, best_score_cebra, best_params_cebra]

        user_results.append(user_row)

        reg_results_df = pd.DataFrame(user_results, columns = ['user', 'best_score_r2', 'best_params'])

        reg_results_path = f"./regression/{modelID_path}"
        auxf.ensure_directory_exists(reg_results_path)

        reg_results_df.to_csv(f"{reg_results_path}/bestmodel_results.csv")

        best_model = grid_search_mlp.best_estimator_
        best_model_list.append(best_model)


        list_gestures = [1, 2, 3, 4, 5, 6 , 7, 8]

        mlp_estimate = best_model.predict(emg_test_transformed)
        # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)


        mlp_estimate_df = pd.DataFrame(mlp_estimate)

        estimates_path = f"{reg_results_path}/estimates"
        auxf.ensure_directory_exists(estimates_path)

        # Save it as a CSV file
        mlp_estimate_df.to_csv(f"{estimates_path}/test_user{user}.csv", index=False)

        DoA_list = [0, 1, 2, 3, 4]

        user_r2_scores = [user]

        for DoA in DoA_list: 
            r2 = r2_score(glove3[:, DoA], mlp_estimate[: , DoA])
            user_r2_scores.append(r2)

        r2_scores.append(user_r2_scores)

        reg_results_df = pd.DataFrame(r2_scores, columns = ['user', 'test_r2_ch0', 'test_r2_ch1', 'test_r2_ch2', 'test_r2_ch3', 'test_r2_ch4'])

        reg_results_df.to_csv(f"{reg_results_path}/test_r2_results.csv")

        for gesture in list_gestures: 
            
            start, end = auxf.cutStimTransition(required_stimulus=gesture, stimulus_data=stim3)

            # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)

            fig, axs = plt.subplots(5, figsize = (10, 10))

            for DoA in DoA_list:

                axs[DoA].plot(np.arange(start, end), glove3[start:end, DoA], label = 'Ground truth', c = 'blue')
                axs[DoA].plot(np.arange(start, end), mlp_estimate[start:end, DoA], label = 'MLP estimate', c = 'red')
                axs[DoA].set_title(f"DoA: {DoA+1}")

                if DoA == 0:
                    axs[DoA].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


            fig.suptitle(f'{modelID_path}: User: {user}, Estimate vs Ground truth for Gesture {gesture} to {gesture+1}') 

            plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust to fit the legend outside the plot
            reg_plot_path = f"{reg_results_path}/user{user}"
            auxf.ensure_directory_exists(reg_plot_path)
            plt.savefig(f"{reg_plot_path}/user{user}_gesture{gesture}_{gesture+1}")
            plt.close(fig)




    return reg_results_path




def runMlpGridsearch_DimRed(dim_red_type: str, perplexity: None, neighbors: None):

    """
    The purpose of this function is to perform dimensionality reduction using ['PCA', 'TSNE', 'UMAP', 'AE_model1', 'AE_model2', 'VAE'] and subseq. perform regression
    using the embeddings of the model by performing gridsearch and cross-validation over MLP architecures. 

    TSNE and VAE are not used for final results
    
    """

    if dim_red_type == 'TSNE' and perplexity is None:
        raise ValueError('You must define a perplexity for TSNE')
    
    elif dim_red_type == 'UMAP' and neighbors is None: 
        raise ValueError('You must define neighbors for UMAP')
    
    user_results = []
    best_model_list = []
    r2_scores = []




    for user in user_list: 

        user_row = []

        emg1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
        emg2 = auxf.getProcessedEMG(user = user, dataset= 2, type_data = 'all')
        emg3 = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')


        stim1 = auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
        stim2 = auxf.getProcessedData(user = user, dataset=2, mode = 'restimulus', rawBool=False)
        stim3 = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)



        glove1 = (auxf.getMappedGlove(user = user, dataset=1)).T
        glove2 = (auxf.getMappedGlove(user = user, dataset=2)).T
        glove3 = (auxf.getMappedGlove(user = user, dataset=3)).T



        # create trainval
        emg_trainval = np.concatenate((emg1, emg2))
        stim_trainval = np.concatenate((stim1, stim2))
        glove_trainval = np.concatenate((glove1, glove2))

        emg_test = emg3

        if dim_red_type == 'TSNE':
            model = TSNE(n_components=3, perplexity=perplexity) #  perplexity is 30 by default
            print('perp:', perplexity)
            modelID = f'TNSE_perp{perplexity}'

        elif dim_red_type == 'PCA':
            model = PCA(n_components=3)
            modelID = 'PCA'

        elif dim_red_type == 'UMAP':
            umap_neighbors = neighbors
            model = UMAP(n_components=3, n_neighbors=umap_neighbors, min_dist=0.001, metric='cosine')
            print('umap_neighbors', umap_neighbors)
            modelID = f'UMAP_neighb{umap_neighbors}'

        elif dim_red_type == 'AE_model1':
            max_epochs = 600
            model = AutoEncoderNet(
                AutoEncoder_1,
                module__num_units=3,
                module__input_size=48,
                lr=0.0001,
                max_epochs=max_epochs,
            )
            modelID = f'AE_epochs{max_epochs}'

        elif dim_red_type == 'AE_model2':
            max_epochs = 850
            model = AutoEncoderNet(
                AutoEncoder_2,
                module__num_units=3,
                module__input_size=48,
                lr=0.00001,
                max_epochs=max_epochs,
            )
            modelID = f'AE_epochs{max_epochs}'
            
        elif dim_red_type == 'VAE':
            model = VariationalAutoEncoderNet(
                VariationalAutoEncoder,
                module__num_units=3,
                module__input_size=48,
                lr=0.0001,
                max_epochs=200,
            )
            modelID = 'VAE'

        else:
            raise ValueError("Please enter valid dimensionality reduction method")


        if dim_red_type not in ['AE', 'VAE']:
            model.fit(emg_trainval)
        else:
            model.fit(emg_trainval.astype(np.float32), emg_trainval.astype(np.float32)) # it needs two inputs but it ignores the second


        if dim_red_type not in ['TSNE', 'AE', 'VAE']:
            emg_trainval_embedding = model.transform(emg_trainval)
            emg_test_embedding = model.transform(emg_test)

        elif dim_red_type == 'TSNE':
            emg_trainval_embedding = model.fit_transform(emg_trainval)
            emg_test_embedding = model.fit_transform(emg_test)

        elif dim_red_type == 'AE':
            _, emg_trainval_embedding = model.forward(emg_trainval.astype(np.float32))
            _, emg_test_embedding = model.forward(emg_test.astype(np.float32))

        # VAE not used in final results

        elif dim_red_type == 'VAE':
            model.module.deterministic = True
            _, emg_trainval_embedding, _, _ = model.forward(emg_trainval.astype(np.float32))
            _, emg_test_embedding, _ ,_ = model.forward(emg_test.astype(np.float32))



        param_grid = {
            'hidden_layer_sizes' : [(256, 256, 256), (256, 256), (150, 150)]
        }


        mlp = MLPRegressor(max_iter = 350)

        grid_search_mlp = GridSearchCV(mlp, param_grid, cv=3, scoring='r2')
        grid_search_mlp.fit(emg_trainval_embedding, glove_trainval)


        best_params_cebra = grid_search_mlp.best_params_
        best_score_cebra = grid_search_mlp.best_score_


        user_row = [user, best_score_cebra, best_params_cebra]

        user_results.append(user_row)

        reg_results_df = pd.DataFrame(user_results, columns = ['user', 'best_score_r2', 'best_params'])

        reg_results_path = f"./regression/{modelID}"
        auxf.ensure_directory_exists(reg_results_path)

        reg_results_df.to_csv(f"{reg_results_path}/bestmodel_results.csv")

        best_model = grid_search_mlp.best_estimator_
        best_model_list.append(best_model)


        list_gestures = [1, 2, 3, 4, 5, 6 , 7, 8]

        mlp_estimate = best_model.predict(emg_test_embedding)
        # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)


        mlp_estimate_df = pd.DataFrame(mlp_estimate)

        estimates_path = f"{reg_results_path}/estimates"
        auxf.ensure_directory_exists(estimates_path)

        # Save it as a CSV file
        mlp_estimate_df.to_csv(f"{estimates_path}/test_user{user}.csv", index=False)

        DoA_list = [0, 1, 2, 3, 4]

        user_r2_scores = [user]

        for DoA in DoA_list: 
            r2 = r2_score(glove3[:, DoA], mlp_estimate[: , DoA])
            user_r2_scores.append(r2)

        r2_scores.append(user_r2_scores)

        reg_results_df = pd.DataFrame(r2_scores, columns = ['user', 'test_r2_ch0', 'test_r2_ch1', 'test_r2_ch2', 'test_r2_ch3', 'test_r2_ch4'])

        reg_results_df.to_csv(f"{reg_results_path}/test_r2_results.csv")

        for gesture in list_gestures: 
            
            start, end = auxf.cutStimTransition(required_stimulus=gesture, stimulus_data=stim3)

            # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)

            fig, axs = plt.subplots(5, figsize = (10, 10))

            for DoA in DoA_list:

                axs[DoA].plot(np.arange(start, end), glove3[start:end, DoA], label = 'Ground truth', c = 'blue')
                axs[DoA].plot(np.arange(start, end), mlp_estimate[start:end, DoA], label = 'MLP estimate', c = 'red')
                axs[DoA].set_title(f"DoA: {DoA+1}")

                if DoA == 0:
                    axs[DoA].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


            fig.suptitle(f'{dim_red_type}: User: {user}, Estimate vs Ground truth for Gesture {gesture} to {gesture+1}') 

            plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust to fit the legend outside the plot
            #plt.legend()
            reg_plot_path = f"{reg_results_path}/user{user}"
            auxf.ensure_directory_exists(reg_plot_path)
            plt.savefig(f"{reg_plot_path}/user{user}_gesture{gesture}_{gesture+1}")
            plt.close(fig)

    
    return reg_results_path




def Dual_CEBRA_MLP_Gridsearch():

    """
    The purpose of this function is to run a gridsearch over CEBRA hyperparameters *and* MLP hyperpameters to perform regression. 
    
    Used for preliminary results but not final results. 
    
    """

    dim_red_type = "MLP_CEBRA_gridsearch"
        
    user_results = []
    best_model_list = []
    r2_scores = []

    for user in user_list:

        print("on user", user)

        emg1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
        emg2 = auxf.getProcessedEMG(user = user, dataset= 2, type_data = 'all')

        stim1 = auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
        stim2 = auxf.getProcessedData(user = user, dataset=2, mode = 'restimulus', rawBool=False)

        glove1 = (auxf.getMappedGlove(user = user, dataset=1)).T
        glove2 = (auxf.getMappedGlove(user = user, dataset=2)).T

        # create trainval
        emg_trainval = np.concatenate((emg1, emg2))
        glove_trainval = np.concatenate((glove1, glove2))

        emg3 = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
        glove3 = (auxf.getMappedGlove(user = user, dataset=3)).T
        stim3 = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)

        iterations = 15000

        pipeline_cebra = Pipeline([
            ('cebra', CEBRA(
                    # model_architecture = 'offset10-model',
                    batch_size= 256,
                    temperature_mode='auto',
                    learning_rate = 0.0001,
                    max_iterations = iterations,
                    output_dimension = 3, 
                    device = "cuda_if_available",
                    verbose = True,
                    conditional='time_delta', 
                    distance = 'cosine',
                )   ), 
            ('mlp', MLPRegressor(max_iter = 350))
        ])



        param_grid = {
            'mlp__hidden_layer_sizes' : [(256, 256, 256), (256, 256), (150, 150)],
            'cebra__model_architecture' : ['offset36-model'],
            'cebra__time_offsets' : [10, 24],
            'cebra__min_temperature' : [0.3, 0.8]
        }

        grid_search_cebra = GridSearchCV(pipeline_cebra, param_grid, cv=3, scoring='r2')

        grid_search_cebra.fit(emg_trainval, glove_trainval)


        best_params_cebra = grid_search_cebra.best_params_
        best_score_cebra = grid_search_cebra.best_score_


        user_row = [user, best_score_cebra, best_params_cebra]

        user_results.append(user_row)

        reg_results_df = pd.DataFrame(user_results, columns = ['user', 'best_score_r2', 'best_params'])

        reg_results_path = f"./regression/{dim_red_type}/MLP"
        auxf.ensure_directory_exists(reg_results_path)

        reg_results_df.to_csv(f"{reg_results_path}/bestmodel_results.csv")

        best_model = grid_search_cebra.best_estimator_
        best_model_list.append(best_model)


        list_gestures = [1, 2, 3, 4, 5, 6 , 7, 8]

        mlp_estimate = best_model.predict(emg3)
        # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)


        mlp_estimate_df = pd.DataFrame(mlp_estimate)

        estimates_path = f"{reg_results_path}/estimates"
        auxf.ensure_directory_exists(estimates_path)

        # Save it as a CSV file
        mlp_estimate_df.to_csv(f"{estimates_path}/test_user{user}.csv", index=False)

        DoA_list = [0, 1, 2, 3, 4]

        user_r2_scores = [user]

        for DoA in DoA_list: 
            r2 = r2_score(glove3[:, DoA], mlp_estimate[: , DoA])
            user_r2_scores.append(r2)

        r2_scores.append(user_r2_scores)

        reg_results_df = pd.DataFrame(r2_scores, columns = ['user', 'test_r2_ch0', 'test_r2_ch1', 'test_r2_ch2', 'test_r2_ch3', 'test_r2_ch4'])

        reg_results_df.to_csv(f"{reg_results_path}/test_r2_results.csv")

        for gesture in list_gestures: 
            
            start, end = auxf.cutStimTransition(required_stimulus=gesture, stimulus_data=stim3)

            # mlp_estimate = auxf.lowpass_filter(data = mlp_estimate, cutoff = 400, fs = 2000)

            fig, axs = plt.subplots(5, figsize = (10, 10))

            for DoA in DoA_list:

                axs[DoA].plot(np.arange(start, end), glove3[start:end, DoA], label = 'Ground truth', c = 'blue')
                axs[DoA].plot(np.arange(start, end), mlp_estimate[start:end, DoA], label = 'MLP estimate', c = 'red')
                axs[DoA].set_title(f"DoA: {DoA+1}")

                if DoA == 0:
                    axs[DoA].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


            fig.suptitle(f'{dim_red_type}: User: {user}, Estimate vs Ground truth for Gesture {gesture} to {gesture+1}') 

            plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust to fit the legend outside the plot
            reg_plot_path = f"{reg_results_path}/user{user}"
            auxf.ensure_directory_exists(reg_plot_path)
            plt.savefig(f"{reg_plot_path}/user{user}_gesture{gesture}_{gesture+1}")
            plt.close(fig)


    path_mlp_pipes = f"./regression/mlp_saved_{dim_red_type}/saved_pipes_mlp"
    auxf.ensure_directory_exists(path_mlp_pipes)

    for index, pipe in enumerate(best_model_list):
        dump(pipe, f"{path_mlp_pipes}/user{index+1}_pipe.joblib")

    return reg_results_path



def runGesturePlot(regression_dir: str):

    """
    The purpose of this function is to generate the plots appertaining to each model. This includes statistics per user, 
    as well as the distribution of test performance across DoA. 
    
    """

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    user_list = np.arange(1, 13)

    
    for user in user_list: 

        r2_scores = pd.read_csv(f"{regression_dir}/test_r2_results.csv")
        r2_scores = r2_scores.drop('Unnamed: 0', axis =1)
        r2_scores = r2_scores.sort_values(by = 'user', ascending=True)

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
        g = g.map(plt.plot, 'actual_time_s', 'Estimate', color='red', label='Estimate')

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
    cv_scores = cv_scores.sort_values(by = 'user')
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






"""

CEBRA-Behaviour and CEBRA-Hybrid are evaluated using a gridsearch over the minimum temperature
and the time_offset, for the 'offset36-model' architecture. The models are loaded from 'saved_models' 
which is generated in model_training_visualisations.py". 

Dimensionality of embeddings kept at 3 for consistency. 

CEBRA-Behaviour:  conditional = 'time_delta', hybrid = False
CEBRA-Hybrid: conditional = 'time_delta', hybrid = False
CEBRA-Time: conditional = 'time', hybrid = False

"""


# CEBRA-Behaviour
mintemp_list = [0.3, 0.5, 0.8]
time_offset_list = [10, 15, 20, 25]

for mintemp in mintemp_list:
    for time_offset in time_offset_list:
        modelID = f'BehContr/offset-36_timeoff{time_offset}_mintemp{mintemp}'
        path = runMlpGridsearch_CEBRA_loaded_models(modelID = modelID)
        print(path)
        runGesturePlot(regression_dir=path)


# CEBRA-Hybrid

mintemp_list = [0.3, 0.5, 0.8]
time_offset_list = [10, 15, 20, 25]

for mintemp in mintemp_list:
    for time_offset in time_offset_list:
        modelID = f'Hybrid/offset-36_timeoff{time_offset}_mintemp{mintemp}'
        path = runMlpGridsearch_CEBRA_loaded_models(modelID = modelID)
        print(path)
        runGesturePlot(regression_dir=path)


# CEBRA-Time tested at optimal hyperparamters for CEBRA-Hybrid

mintemp_list = [0.5]
time_offset_list = [10]

for mintemp in mintemp_list:
    for time_offset in time_offset_list:
        modelID = f'SelfSup/offset-36_timeoff{time_offset}_mintemp{mintemp}'
        path = runMlpGridsearch_CEBRA_loaded_models(modelID = modelID)
        runGesturePlot(regression_dir=path)






"""

UMAP is tested at 40, 75 and 100 neighbours. Minimum distance is kept at 0.001, n_components = 3 (consistent across all methods)
and distance metric used is cosine

"""

UMAP_neighbours = [40, 75, 100]

for neighbours in UMAP_neighbours:
    path = runMlpGridsearch_DimRed(dim_red_type="UMAP", perplexity=None, neighbors=neighbours) 
    runGesturePlot(regression_dir=path)


"""

PCA - no hyperparameters to tune. 3 Principal Components for uniformity across all dimensionality reduction methods 

"""

path = runMlpGridsearch_DimRed(dim_red_type="PCA", perplexity=None, neighbors=neighbours) 
runGesturePlot(regression_dir=path)


"""

Autoencoder: 2 Autoencoder architectures are used and their model architecture is defined in autoencoders_skorch.py 
['Autoencoder_1', 'Autoencoder_2']

"""

path = runMlpGridsearch_DimRed(dim_red_type="AE_model1", perplexity=None, neighbors=neighbours) 
runGesturePlot(regression_dir=path)

path = runMlpGridsearch_DimRed(dim_red_type="AE_model2", perplexity=None, neighbors=neighbours) 
runGesturePlot(regression_dir=path)
