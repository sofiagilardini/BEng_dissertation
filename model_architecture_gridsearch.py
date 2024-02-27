import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import datapipe_redu as dtp
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

plt.rcParams.update(
    {
        "lines.markersize": 20,  # Big points
        "font.size": 15,  # Larger font
        "xtick.major.size": 10.0,  # Bigger xticks
        "ytick.major.size": 10.0,  # Bigger yticks
    }
)


## this should be automated with auxf.getProcessedData ** TO DO ***

# # using the validation dataset for hyperparameter tuning (grid search)
# emg_data = np.load("/home/sofia/beng_thesis/validation_data/emg_data_processed/emg_1_2_450_128_1_0102.npy")
# glove_data = np.load("/home/sofia/beng_thesis/validation_data/glove_data_processed/glove_1_2_128_1.npy")
# restimulus_data = np.load("/home/sofia/beng_thesis/validation_data/restimulus_data_processed/restimulus_1_2_128_1.npy")

emg_data = np.load("/home/sofia/beng_thesis/training_data/emg_data_processed/emg_1_1_450_128_52_0102.npy")
glove_data = np.load("training_data/glove_data_processed/glove_1_1_128_52.npy")
restimulus_data = np.load("training_data/restimulus_data_processed/restimulus_1_1_128_52.npy")


print(emg_data.shape)
print(glove_data.shape)
print(restimulus_data.shape)

# taking a portion as the dataset is very large
split_gridsearch = emg_data.shape[0] // 2

emg_valid = emg_data[:split_gridsearch]
glove_valid = glove_data[:split_gridsearch]
restim_valid = restimulus_data[:split_gridsearch]

figsize = (30, 20)

grid_search = cebra.grid_search.GridSearch()


@cebra.models.register("my-model-1") 
class MyModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 40),
            nn.GELU(),
            nn.Conv1d(num_units, num_output, 5),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(22, 23)


# @cebra.models.register("my-model-2") 
# class MyModel(_OffsetModel, ConvolutionalModelMixin):

#     def __init__(self, num_neurons, num_units, num_output, normalize=True):
#         super().__init__(
#             nn.Conv1d(num_neurons, num_units, 2),
#             nn.GELU(),
#             nn.Conv1d(num_units, num_units, 40),
#             nn.GELU(),
#             nn.Conv1d(num_units, num_output, 5),
#             num_input=num_neurons,
#             num_output=num_output,
#             normalize=normalize,
#         )


#     # ... and you can also redefine the forward method,
#     # as you would for a typical pytorch model

#     def get_offset(self):
#         return cebra.data.Offset(22, 23)


@cebra.models.register("my-model-2") 
class MyModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.BatchNorm1d(num_units),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 40),
            nn.BatchNorm1d(num_units),
            nn.GELU(),
            nn.Conv1d(num_units, num_output, 5),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
            )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(22, 23)


# @cebra.models.register("my-model-4") 
# class MyModel(_OffsetModel, ConvolutionalModelMixin):

#     def __init__(self, num_neurons, num_units, num_output, normalize=True):
#         super().__init__(
#             nn.Conv1d(num_neurons, num_units, 2),
#             nn.GELU(),
#             nn.Conv1d(num_units, num_units, 40),
#             nn.GELU(),
#             nn.Conv1d(num_units, num_output, 5),
#             num_input=num_neurons,
#             num_output=num_output,
#             normalize=normalize,
#         )

#     # ... and you can also redefine the forward method,
#     # as you would for a typical pytorch model

#     def get_offset(self):
#         return cebra.data.Offset(22, 23)


# @cebra.models.register("my-model-5") 
# class MyModel(_OffsetModel, ConvolutionalModelMixin):

#     def __init__(self, num_neurons, num_units, num_output):
#         super().__init__(
#         nn.Conv1d(num_neurons, 2 * num_units, 2),
#         nn.GELU(),
#         nn.Conv1d(2 * num_units, 2 * num_units, 40),
#         nn.GELU(),
#         nn.Conv1d(2 * num_units, num_output, 5),
#         )

#     # ... and you can also redefine the forward method,
#     # as you would for a typical pytorch model

#     def get_offset(self):
#         return cebra.data.Offset(22, 23)



def gridSearch(model: str, iterations: int, training: str):

    dataset_class = {"dataset_classification": (emg_valid, restim_valid)}     # classification
    dataset_regression = {"dataset_regression": (emg_valid, glove_valid)}     # regression

    # Define the parameters, either variable or fixed
    params_grid = dict(
        model_architecture = model,
        batch_size = [2**8],
        temperature_mode='auto',
        #min_temperature = 0.9,
        learning_rate = [0.001, 0.0001],
        max_iterations = iterations,
        time_offsets = [5],
        output_dimension = [9, 12],
        device = "cuda_if_available",
        verbose = True,
        conditional='time_delta',
        distance = 'cosine',
        
        )
    

    if training == 'C':
        datasets = dataset_class
        models_dir = f"./model_architecture_gridsearch/saved_models_classification/{model}"
        fig_path = f"./model_architecture_gridsearch/loss_plots/{model}/classification.png"
        resultspath = f"./model_architecture_gridsearch/saved_models_classification/grid_search_results"

    elif training == 'R':
        datasets = dataset_regression
        models_dir = f"./model_architecture_gridsearch/saved_models_regression/{model}"
        fig_path = f"./model_architecture_gridsearch/loss_plots/{model}/regression.png"
        resultspath = f"./model_architecture_gridsearch/saved_models_regression/grid_search_results"


    else:
        print("classification nor regression label given, ERROR")
        
        return
    

    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= models_dir)

    df_results = grid_search.get_df_results()
    df_results.to_csv(f'{resultspath}/grid_search_results_{model}.csv', index=False)

    
    ax = grid_search.plot_loss_comparison(figsize = figsize)
    plt.savefig(fig_path)


# def offset10_run_class():

#     model = "offset10-model"

#     # Define the parameters, either variable or fixed
#     params_grid = dict(
#         model_architecture = "offset10-model",
#         batch_size = 128,
#         temperature_mode='auto',
#         learning_rate = [0.1, 0.01, 0.001],
#         max_iterations = iterations,
#         time_offsets = [5, 20],
#         output_dimension = [3, 6, 12, 15],
#         device = "cuda_if_available",
#         verbose = True,
#         conditional='time_delta',
#         distance = 'cosine',
        
#         )

                            

#     # 3. Create and fit the grid search to your data
#     grid_search = cebra.grid_search.GridSearch()
#     grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

#     df_results = grid_search.get_df_results()
#     df_results.to_csv('grid_search_results.csv', index=False)



#     # Now plot using grid_search.plot_loss_comparison()
#     ax = grid_search.plot_loss_comparison(figsize = figsize)
#     plt.savefig(f"./loss_plots/{model}/classification.png")


# def mymodel1run_class():

#     model = 'my-model-1'

#     # Define the parameters, either variable or fixed
#     params_grid = dict(
#         model_architecture = model,
#         batch_size = 256,
#         temperature_mode='auto',
#         learning_rate = [0.1, 0.01, 0.001],
#         max_iterations = iterations,
#         time_offsets = [5, 10, 20],
#         output_dimension = [3, 6, 12, 15],
#         device = "cuda_if_available",
#         verbose = True,
#         conditional='time_delta',
#         distance = 'cosine',
        
#         )



#     grid_search = cebra.grid_search.GridSearch()
#     grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

#     df_results = grid_search.get_df_results()
#     df_results.to_csv('grid_search_results.csv', index=False)



#     # Now plot using grid_search.plot_loss_comparison()
#     ax = grid_search.plot_loss_comparison(figsize = figsize)
#     plt.savefig(f"./loss_plots/{model}/classification.png")


# def offset1mse_class():

#     model = 'offset1-model-mse'

#     # Define the parameters, either variable or fixed
#     params_grid = dict(
#         model_architecture = model,
#         batch_size = 256,
#         temperature_mode='auto',
#         learning_rate = [0.1, 0.01, 0.001],
#         max_iterations = iterations,
#         time_offsets = [5, 10, 20],
#         output_dimension = [3, 6, 12, 15],
#         device = "cuda_if_available",
#         verbose = True,
#         conditional='time_delta',
#         distance = 'cosine',
        
#         )


#     grid_search = cebra.grid_search.GridSearch()
#     grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

#     df_results = grid_search.get_df_results()
#     df_results.to_csv('grid_search_results.csv', index=False)



#     # Now plot using grid_search.plot_loss_comparison()
#     ax = grid_search.plot_loss_comparison(figsize = figsize)
#     plt.savefig(f"./loss_plots/{model}/classification.png")





# ready to run ! :) 


# get best results from classification and regression

def getBestModels(training: str, n: int):

    if training == 'C':
        resultspath = f"./model_architecture_gridsearch/saved_models_classification/grid_search_results"
        resultsdfpath = f"./model_architecture_gridsearch/saved_models_classification/combined_results_{training}.csv"
        bestmodelspath = f"./model_architecture_gridsearch/saved_models_classification/best_models/{training}-df.csv"


    if training == 'R':
        resultspath = f"./model_architecture_gridsearch/saved_models_regression/grid_search_results"
        resultsdfpath = f"./model_architecture_gridsearch/saved_models_regression/combined_results_{training}.csv"
        bestmodelspath = f"./model_architecture_gridsearch/saved_models_regression/best_models/{training}-df.csv"



    dataframes_list = []

    for filename in os.listdir(resultspath):

        model_name = filename.split('_')[-1].split('.csv')[0]

        print(model_name)

        file_path = os.path.join(resultspath, filename)

        if os.path.isfile(file_path) and filename.endswith(".csv"):

            df = pd.read_csv(file_path)

            df['model'] = model_name

            dataframes_list.append(df)


    combined_dataframe = pd.concat(dataframes_list, ignore_index=True)

    combined_dataframe = combined_dataframe.sort_values(by = 'loss', ascending = True)

    combined_dataframe.to_csv(resultsdfpath, index = False)

    # get the n best models 

    best_models_df = combined_dataframe.head(n)

    best_models_df.to_csv(bestmodelspath, index = False)



# once auxf.getProcessedData is done this will work -
# def trainBestModels(user: int, dataset: int, training: str, iterations: int):

## temporary func def: 

def trainBestModels(training: str, iterations: int, training_bool: bool):


    ### NEED AUXF.GETPROCESSEDDATA() *******


    # # using the TRAINING dataset (user = 1, dataset = 1)
    # emg_data = np.load("./training_data/emg_data_processed/emg_1_1_450_128_1_0102.npy")
    # glove_data = np.load("./training_data/glove_data_processed/glove_1_1_128_1.npy")
    # restimulus_data = np.load("./training_data/restimulus_data_processed/restimulus_1_1_128_1.npy")

    # train the best 10 models on the entire dataset for user 1 dataset 1 (**need to check training vs validation**)

    print(emg_data.shape)

    if training == "C":
        bestmodelspath = f"./model_architecture_gridsearch/saved_models_classification/best_models"


    if training == "R":
        bestmodelspath = f"./model_architecture_gridsearch/saved_models_regression/best_models"
    
    
    best_models_df = pd.read_csv(f"{bestmodelspath}/{training}-df.csv")

    cebra_models = []
    labels = []
    paths = []

    # iterate through all the models in the best models dataframe and train on the entire dataset
    for row in range(len(best_models_df)):

        # build the model based on the paramgrid

        model = best_models_df['model'][row]
        #batch_size = best_models_df['batch_size'][row]
        learning_rate = best_models_df['learning_rate'][row]
        #time_offsets = best_models_df['time_offsets'][row]
        output_dimension = best_models_df['output_dimension'][row]

        cebra_model = CEBRA(
            model_architecture = model,
            #batch_size = batch_size,
            batch_size= 128,
            temperature_mode='auto',
            learning_rate = learning_rate,
            max_iterations = iterations,
            time_offsets=5,
            #time_offsets = time_offsets,
            output_dimension = output_dimension,
            device = "cuda_if_available",
            verbose = True,
            conditional='time_delta',
            distance = 'cosine',
        )

        if training_bool == True:

            if training == 'C':
                #cebra_models[row].fit(emg_data, restimulus_data)
                cebra_model.fit(emg_data, restimulus_data)


            if training == 'R':
                #cebra_models[row].fit(emg_data, glove_data)
                cebra_model.fit(emg_data, glove_data)

        #label = f"{model}_{batch_size}_{learning_rate}_{time_offsets}_{output_dimension}"

        label = 'test'

        path = f"{bestmodelspath}/best-{label}.pt"


        paths.append(path)
        labels.append(label)

        if training_bool == True:
        #cebra_models[row].save(f"{bestmodelspath}/row-{row}.pt")
            cebra_model.save(path)
            cebra_models.append(cebra_model)
            ax = cebra.compare_models([cebra_models[i] for i in range(len(cebra_models))], labels=labels, figsize=figsize)




    plt.savefig(f"{bestmodelspath}/loss_comparison")

    return paths



iterations = 40000


# # # classification

list_models = [1, 2, 3, 4, 5]

list_models = [1, 2]


# for num in list_models:
#     gridSearch(model = f'my-model-{num}', iterations = iterations, training = 'C')


gridSearch(model = 'offset10-model', iterations = iterations, training = 'C')
#gridSearch(model = 'my-model-2', iterations = iterations, training = 'R')

# # regression

# gridSearch(model = "offset10-model", iterations = iterations, training = 'R')
# gridSearch(model = 'my-model-1', iterations = iterations, training = 'R')
# gridSearch(model = 'offset1-model-mse', iterations = iterations, training = 'R')


#from the grid search, get the 'n' models with the lowest loss

getBestModels("C", n =3)
getBestModels("R", n = 5)


# train the n best models on the entire training dataset 
#best_models_C = trainBestModels(training = 'C', iterations= 1500, training_bool=False) # mse is messing everything up (removed)
#best_models_R = trainBestModels(training = 'R', iterations = 5, training_bool=False)


# cebra_model = CEBRA(
#     model_architecture = "offset10-model",
#     batch_size = 1012,
#     temperature_mode="auto",
#     learning_rate = 0.01,
#     max_iterations = 2 ,
#     time_offsets = 1,
#     output_dimension = 15,
#     device = "cuda_if_available",
#     verbose = True
# )


# cebra_model2 = CEBRA(
#     model_architecture = "offset10-model",
#     batch_size = 512,
#     temperature_mode="auto",
#     learning_rate = 0.01,
#     max_iterations = 2 ,
#     time_offsets = 1,
#     output_dimension = 6,
#     device = "cuda_if_available",
#     verbose = True
# )



# cebra_model = cebra.CEBRA.load("/home/sofia/beng_thesis/saved_models_classification/best_models/best-offset10-model_128_0.001_5_15.pt")
# cebra_model2 = cebra.CEBRA.load()



# tr_emg_data = np.load("./training_data/emg_data_processed/emg_1_1_450_128_1_0102.npy")
# tr_glove_data = np.load("./training_data/glove_data_processed/glove_1_1_128_1.npy")
# tr_restimulus_data = np.load("./training_data/restimulus_data_processed/restimulus_1_1_128_1.npy")

# # using the validation dataset for hyperparameter tuning (grid search)
# emg_data = np.load("/home/sofia/beng_thesis/validation_data/emg_data_processed/emg_1_2_450_128_1_0102.npy")
# glove_data = np.load("/home/sofia/beng_thesis/validation_data/glove_data_processed/glove_1_2_128_1.npy")
# restimulus_data = np.load("/home/sofia/beng_thesis/validation_data/restimulus_data_processed/restimulus_1_2_128_1.npy")

# train = 75000
# test = 5000

# model_list = best_models_C

# print(model_list)

# k_list = [1, 5, 10]

# # print(restimulus_data[train:train+test])

# # # why am I getting a 'continuous' error? 

# # # also hwo to load model ?

# restimulus_data = restimulus_data.astype(int).flatten()
# tr_restimulus_data = tr_restimulus_data.astype(int).flatten()

# len_tr = len(restimulus_data)//9

# # auxf.KNN_heatmap(model_list, emg_data[:train], emg_data[train: train+test], restimulus_data[:train], restimulus_data[train:train+test], k_list)

# auxf.KNN_heatmap(model_list, tr_emg_data[:len_tr], emg_data[:len_tr], tr_restimulus_data[:len_tr], restimulus_data[:len_tr], k_list)


# # #model = cebra.load_model("./saved_models_classification/offset10-model/batch_size_128_learning_rate_0.1_output_dimension_3_time_offsets_1_dataset_classification.pt")
    
