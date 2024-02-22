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




# using the validation dataset for hyperparameter tuning (grid search)
emg_data = np.load("/home/sofia/beng_thesis/validation_data/emg_data_processed/emg_1_2_450_256_1_0102.npy")
glove_data = np.load("/home/sofia/beng_thesis/validation_data/glove_data_processed/glove_1_2_256_1.npy")
restimulus_data = np.load("/home/sofia/beng_thesis/validation_data/restimulus_data_processed/restimulus_1_2_256_1.npy")

print(emg_data.shape)
print(glove_data.shape)
print(restimulus_data.shape)

# taking a portion as the dataset is very large
split_gridsearch = emg_data.shape[0] // 2

emg_valid = emg_data[:split_gridsearch]
glove_valid = glove_data[:split_gridsearch]
restim_valid = restimulus_data[:split_gridsearch]

figsize = (30, 20)



# cebra_model1 = CEBRA(
#     model_architecture = "my-model",
#     batch_size = 256,
#     temperature_mode='auto',
#     #temperature= 0.9,
#     learning_rate = 0.1,
#     max_iterations = 1000,
#     time_offsets = 4000,
#     output_dimension = 3,
#     device = "cuda_if_available",
#     verbose = True,
#     conditional='time_delta',
#     distance = 'cosine',
# )

@cebra.models.register("my-model-1") # --> add that line to register the model!
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


@cebra.models.register("my-emg-model")
class MyEMGModel(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, kernel_size=3, stride=1),
            nn.GELU(),
            nn.Conv1d(num_units, num_units*2, kernel_size=5, stride=1),
            nn.GELU(),
            nn.Conv1d(num_units*2, num_units*4, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Conv1d(num_units*4, num_output, kernel_size=7, stride=2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    
    def get_offset(self):
        return cebra.data.Offset(22, 23)
    


datasets = {"dataset_classification": (emg_valid, glove_valid, restim_valid)}     # time contrastive learning



def offset10_run():

    model = "offset10-model"

    # Define the parameters, either variable or fixed
    params_grid = dict(
        #model_architecture = ["my-model-1", "my-emg-model", "offset10-model"],
        model_architecture = "offset10-model",
        batch_size = 256,
        temperature_mode='auto',
        learning_rate = [0.1, 0.01, 0.001],
        max_iterations = iterations,
        time_offsets = [5, 20],
        output_dimension = [3, 6, 12, 15],
        device = "cuda_if_available",
        verbose = True,
        conditional='time_delta',
        distance = 'cosine',
        
        )

                            

    # 3. Create and fit the grid search to your data
    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

    df_results = grid_search.get_df_results()
    df_results.to_csv('grid_search_results.csv', index=False)



    # Now plot using grid_search.plot_loss_comparison()
    ax = grid_search.plot_loss_comparison(figsize = figsize)
    plt.savefig(f"./loss_plots/{model}/classification.png")




def mymodel1run():

    model = 'my-model-1'

    # Define the parameters, either variable or fixed
    params_grid = dict(
        model_architecture = model,
        batch_size = 256,
        temperature_mode='auto',
        learning_rate = [0.1, 0.01, 0.001],
        max_iterations = iterations,
        time_offsets = [5, 10, 20],
        output_dimension = [3, 6, 12, 15],
        device = "cuda_if_available",
        verbose = True,
        conditional='time_delta',
        distance = 'cosine',
        
        )



    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

    df_results = grid_search.get_df_results()
    df_results.to_csv('grid_search_results.csv', index=False)



    # Now plot using grid_search.plot_loss_comparison()
    ax = grid_search.plot_loss_comparison(figsize = figsize)
    plt.savefig(f"./loss_plots/{model}/classification.png")





def offset1mse():

    model = 'offset1-model-mse'

    # Define the parameters, either variable or fixed
    params_grid = dict(
        model_architecture = model,
        batch_size = 256,
        temperature_mode='auto',
        learning_rate = [0.1, 0.01, 0.001],
        max_iterations = iterations,
        time_offsets = [5, 10, 20],
        output_dimension = [3, 6, 12, 15],
        device = "cuda_if_available",
        verbose = True,
        conditional='time_delta',
        distance = 'cosine',
        
        )


    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= f"saved_models_classification/{model}")

    df_results = grid_search.get_df_results()
    df_results.to_csv('grid_search_results.csv', index=False)



    # Now plot using grid_search.plot_loss_comparison()
    ax = grid_search.plot_loss_comparison(figsize = figsize)
    plt.savefig(f"./loss_plots/{model}/classification.png")




iterations = 20


mymodel1run()
offset10_run()
offset1mse()





plt.show()