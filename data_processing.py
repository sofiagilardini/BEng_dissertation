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



# winsize = 256ms, stride length = 52ms (about 60% overlap), cutoff freq = 450Hz

fs = 2000 # 
order = 1
win_size = 128
win_stride = 52
cutoff_f = 450


list_users = [1, 2, 3, 4, 5, 6, 7, 10]
# feat_ID_list = ['raw', 'all', "RMS"]
feat_ID_list = ['raw']

dataset_list = [1, 2, 3]
rawBool_list = [True, False]



def runEMG():
    for user in list_users:
        for feat_ID in feat_ID_list:
            for dataset in dataset_list:

                print('user in processing', user)

                auxf.emg_process(cutoff_val = cutoff_f, size_val = win_size, stride_val = win_stride, user = user, dataset=dataset, order = order, feat_ID=feat_ID)


def runRest():
    for rawBool in rawBool_list:
        for user in list_users:
            for dataset in dataset_list:
                print('user in processing', user)

                auxf.glove_process(size_val = win_size, stride_val = win_stride, user = user, dataset = dataset, rawBool = rawBool)
                auxf.restimulusProcess(size_val= win_size, stride_val= win_stride, user = user, dataset= dataset, rawBool = rawBool)






# perform quick check on the data running models 
# this should throw no errors - ensuring downstream is clean
                
            

model = CEBRA(
                model_architecture = 'offset10-model',
                batch_size= 64,
                temperature_mode='auto',
                learning_rate = 0.0001,
                max_iterations = 10,
                min_temperature=1.2,
                time_offsets = 25,
                output_dimension = 3, 
                device = "cuda_if_available",
                verbose = True,
                conditional='time_delta',
                distance = 'cosine' 
            )   


directory = './processed_data'



for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        data = np.load(file_path)

        if file_path.__contains__("5"):
            continue # user 5 has NaN errors in the glove data 

        print("fitting on", file_path)
        print("shape", data.shape)

        model.fit(data)