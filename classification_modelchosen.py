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
from copy import deepcopy



iterations = 10

# ### ---- DEFINE MODEL ------- ###

cebra_model_def = CEBRA(
    model_architecture = 'offset10-model',
    #batch_size = batch_size,
    batch_size= 256,
    temperature_mode='auto',
    learning_rate = 0.0001,
    max_iterations = iterations,
    time_offsets=5,
    #time_offsets = time_offsets,
    output_dimension = 6,
    device = "cuda_if_available",
    verbose = True,
    conditional='time_delta',
    distance = 'cosine',
)




# ### ---- DEFINE MODEL ------- ###


num_users = 13
k_list = [5, 7, 9]
k_list_str = [str(k) for k in k_list]

print(k_list_str)

KNN_df = pd.DataFrame(columns = k_list_str)

#model = cebra.CEBRA.load("model_architecture_gridsearch/saved_models_classification/offset10-model/learning_rate_0.001_output_dimension_12_dataset_classification.pt")

#for user in range(num_users):

for user in range(3):

    user = user + 1

    print(user)


    emg_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'emg')
    emg_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'emg')
    emg_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'emg')

    glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'glove')
    glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'glove')
    glove_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'glove')
    
    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'restimulus').astype(int)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'restimulus').astype(int)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)

    cebra_model = deepcopy(cebra_model_def)

    cebra_model.fit(emg_tr1)

    mitigationslice = 20

    print(emg_tr1.shape)

    heatmap_df = auxf.KNN_heatmap_df(model = cebra_model, 
                                     emg_train = emg_tr1[:len(emg_tr1)-mitigationslice], 
                                     emg_test = emg_test[:len(emg_tr1)-mitigationslice], 
                                     rest_train = restim_tr1[:len(emg_tr1)-mitigationslice],
                                     rest_test= restim_test[:len(emg_tr1)-mitigationslice],
                                     k_list = k_list, 
                                     user = user)
    

    KNN_df = pd.concat([KNN_df, heatmap_df], ignore_index=True)


KNN_df.to_csv('./classification_results/KNN_classification_results.csv', index = True)



# emg_tr1 = auxf.getProcessedData(user = 2, dataset = 1, type_data = 'training', mode = 'emg')

# df = pd.DataFrame(data = emg_tr1)


# print(df.isna())

# inf_indices = np.where(np.isinf(df))

# # Convert indices to DataFrame indices and columns
# inf_rows = df.index[inf_indices[0]]
# inf_cols = df.columns[inf_indices[1]]

# # Combine rows and columns for exact locations of inf values
# inf_locations = list(zip(inf_rows, inf_cols))

# print(inf_locations)

# print(len(df))

# emg_tr2 = auxf.getProcessedData(user = 2, dataset = 1, type_data = 'training', mode = 'emg')


# df = pd.DataFrame(data = emg_tr2)

# #df = df.iloc[:-15]

# print(df.isna())

# inf_indices = np.where(np.isinf(df))

# # Convert indices to DataFrame indices and columns
# inf_rows = df.index[inf_indices[0]]
# inf_cols = df.columns[inf_indices[1]]

# # Combine rows and columns for exact locations of inf values
# inf_locations = list(zip(inf_rows, inf_cols))

# print(inf_locations)

# print(len(df))