import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import datapipe as dtp
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
#import pickle





def emg_process(cutoff_val, size_val, stride_val, user, dataset):


        size, stride = auxf.slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        user = 1
        dataset = 1

        # 2. get the data: input is user and which dataset (dataset should only be 1 or 2)

        emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = auxf.getdata(user = user, dataset = dataset)
        print("got emg data")

        # 3. low-pass filter the EMG data and extract features

        for i in range(emg_u1_d1.shape[1]):
            emg_u1_d1[:, i] = auxf.lowpass_filter(emg_u1_d1[:, i], cutoff_val, fs, order)
            print("doing low pass filter at step ", i)


        emg_windows = []

        # Loop through each channelc
        num_channelsEMG = emg_u1_d1.shape[1] 


        # low pass and features = [WL and LV] -> eventually you should be able to choose these
        for i in range(num_channelsEMG):
            emg_window = auxf.slidingWindowEMG(emg_u1_d1[:, i], size, stride)
            print("emg windowing at step ", i)
            emg_window_WL = auxf.WL(emg_window, size, stride) 
            print("emg WL extracting at step ", i)
            WL = True
            emg_window_LV = auxf.LV(emg_window, size)
            print("emg LV extracting at step ", i)
            LV = True
            emg_windows.append(emg_window_WL)
            emg_windows.append(emg_window_LV)


        emg_windows_stacked = np.array(emg_windows)
        emg_windows_stacked = np.transpose(emg_windows_stacked)

        if WL == True and LV == True:
            feat_ID = '0102'

        # I need to save this in a folder named EMG data, processed 
        
        print("about to save EMG data")
        np.save(f"./emg_data_processed/emg_{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}.npy", emg_windows_stacked)
        print("saved EMG data")


        emg_data_ID = f"{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}"

        return emg_data_ID




def glove_process(cutoff_val, size_val, stride_val, user, dataset):
     
    size, stride = auxf.slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))
    
    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = auxf.getdata(user = user, dataset = dataset)
    print("got glove data")


    # low pass the glove data ? no ? glove data should not be that noisy

    # for i in range(glove_u1_d1.shape[1]):
    #     glove_u1_d1[:, i] = auxf.lowpass_filter(glove_u1_d1[:, i], cutoff_val, fs, order)
     

    # perform sliding windows on the data
        
    glove_data = []

    for i in range(glove_u1_d1.shape[1]):
        glove_data.append(auxf.slidingWindowGlove(glove_u1_d1[:, i], size, stride))
        print("doing sliding window on glove at step", i)

    glove_data_array = np.array(glove_data)

    np.save(f"./glove_data_processed/glove_{user}_{dataset}_{size_val}_{stride_val}.npy", glove_data_array)
    print("saved glove data")

    glove_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

    return glove_data_ID


def cebraGridSearch_Multi(emg_data, emg_data_ID, glove_data, glove_data_ID):


    models_dir = f"saved_models/{emg_data_ID}_{glove_data_ID}"

    # 1. Define the parameters, either variable or fixed
    params_grid = dict(
        model_architecture = ["offset5-model", "offset10-model"], # I need to define my own models here 
        batch_size = [2**5, 2**6, 2**7],
        output_dimension = [3, 6, 9],
        learning_rate = [0.001, 0.0001],
        time_offsets = [1, 5, 10, 15, 20, 50],
        max_iterations = 1, #10000,
        temperature_mode = "auto",
        verbose = True)


    # set fraction for training and for testing 

    training_ratio = 0.4
    testing_ratio = 0.2

    # not sure if it is 0 or 1

    trainingset = emg_data.shape[0] * training_ratio
    testingset = emg_data.shape[0] * testing_ratio


    glove_channels_list = [1, 4, 6, 8, 11, 14]


    # 2. Define the datasets to iterate over
    datasets = {"dataset1": (emg_data[:trainingset, :], glove_data[:trainingset, :])} # behavioral contrastive learning all glove channels 
                #"dataset2": (emg_data[:trainingset, :], glove_data[:trainingset, glove_channels_list])} # behavioral contrastive learning selection of channels (zhangcontinuous)


    # 3. Create and fit the grid search to your data
    grid_search = cebra.grid_search.GridSearch()


    # fit the model and then save in a folder that is specified according to the emg_data_ID and glove_data_ID -> although I am not sure if the 
    # find best model function is going to work in that case

    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir= models_dir)

    # 4. Get the results
    df_results = grid_search.get_df_results(models_dir= models_dir)

    df_results.to_csv("df_models.csv")

    # save dictionary to person_data.pkl file

    # here, instead of getting the best model for saved_models, I could iterate through every sub folder of saved_models and find the best model there ?
    models, parameter_grid = grid_search.load(dir=models_dir)


    # 5. Get the best model for a given dataset
    best_model, best_model_name = grid_search.get_best_model(dataset_name="dataset1", models_dir=models_dir)

    print(f"name of best model for emg {emg_data_ID} and glove {glove_data_ID}", best_model_name)

    return best_model, best_model_name



cutoff_emg = [200, 300, 400, 450] # Hz
win_size = [150, 200, 250, 300, 350, 400] # ms
win_stride = [20, 25, 30, 40] # ms 

# these don't change for now 
user = 1
dataset = 1

fs = 2000 # sampling f (Hz)
order = 6 # order for LPF


for cutoff in cutoff_emg:
    for size in win_size:
        for stride in win_stride:
            emg_process(cutoff_val=cutoff, size_val=size, stride_val=stride, user = user, dataset = dataset)

for size in win_size:
    for stride in win_stride:
        glove_process(cutoff_val = None, size_val = size, stride_val = stride, user = user, dataset = dataset)



# def extractIDs(directory):

#     emg_id_list = []
#     glove_id_list = []

#     for filename in os.listdir(directory):
#         if filename.endswith('.npy'):

#             # if emg data
#             if filename.startswith('emg'):
#                 parts = filename.replace('.npy', "").split("_")
#                 type_data, user, dataset, cutoff_val, size_val, stride_val, feat_ID = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
#                 emg_data_ID = f"{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}"
#                 emg_id_list.append(emg_data_ID)

#                 glove_id_list = None
            

#             # if glove data
#             if filename.startswith('glove'):
#                 parts = filename.replace('.npy', "").split("_")
#                 type_data, user, dataset, size_val, stride_val = parts[0], parts[1], parts[2], parts[3], parts[4]
#                 glove_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"
#                 glove_id_list.append(glove_data_ID)

#                 emg_id_list = None
    
#     if emg_id_list == None:
#         return glove_id_list
    
#     elif glove_id_list == None:
#         return emg_id_list


# def findBestModel(emg_ID, glove_ID, directory):

#     emg_data_in = f"./emg_data_processed/emg_{emg_ID}.npy"
#     glove_data_in = f"glove_data_processed/glove_{gloveID}.npy"

#     # cebraGridSearch_Multi returns best_model, best_model_name


#     best_model, best_model_name = cebraGridSearch_Multi(emg_data=emg_data_in,
#                           glove_data=glove_data_in,
#                           emg_data_ID=emg_ID,
#                           glove_data_ID=glove_ID)
    


#     # give the best model an ID and then save it in the folder best_models

