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

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor



list_models = []
dim = 3 

directory_loadmodels = "./results_generation/model_training/behaviour_contrastive"



# load all regression models
for dirpath, dirnames, filenames in os.walk(directory_loadmodels):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        
        if file_path.__contains__(".pt"):

            model_path = file_path
            list_models.append(model_path)
            print(model_path)



def runRegression(model_path, reg_type: str, gesture, results_dir: str):


    """
    reg_type: ["MLP", "GB", "PLSR"]
    
    """

    gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]

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


    glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='glove', rawBool = rawBool)
    glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='glove', rawBool = rawBool)
    glove_test = auxf.getProcessedData(user = user, dataset = 3, mode='glove', rawBool = rawBool)

    emg_tr_concat = np.concatenate([emg_tr1, emg_tr2])
    glove_tr_concat = np.concatenate([glove_tr1, glove_tr2])


    embedding_tr = model.transform(emg_tr_concat)

    # this is the embedding that then I need to plot
    embedding_test = model.transform(emg_test)


    # this selects the test embedding based on which gesture transition we are on 

    start, end = auxf.cutStimTransition(restim_test, gesture) 
    embedding_test = embedding_test[start:end, :]


    if reg_type == "MLP":
        reg = MLPRegressor((256, 256)) # to review - layers

    if reg_type == "GB":
        reg = GradientBoostingRegressor(n_estimators=100) # to review - estimators

    if reg_type == "PSLR":
        raise ValueError("Not implemented yet, use MLP or GB")
    

    # this is just the accuracy scores for all the gestures for one glove channel -> although now I think it should be the other way around. 
    # i.e one row of data in the df has all the glove channel accuracies for one gesture (not one row has all the gesture accuracies for one glove state)
    # but will leave for now with 9th channel (need to figure out)


    channels_accuracy_list = []
    predictions_gesture = []

    channels = np.arange(0, 18, 1)

    for i, channel in enumerate(channels):

        reg.fit(embedding_tr, glove_tr_concat[:, channel])

        reg_pred = reg.predict(embedding_test)
        reg_pred = reg_pred.tolist()
        predictions_gesture.append(reg_pred)

        accuracy_score = reg.score(embedding_test, glove_test[start:end, channel])

        channels_accuracy_list.append(accuracy_score)

    
    predictions_gesture = pd.DataFrame(predictions_gesture)
    predictions_gesture_path = f"{results_dir}/regression_predictions"
    auxf.ensure_directory_exists(predictions_gesture_path)
    predictions_gesture.to_csv(f"{predictions_gesture_path}/{model_ID}_{reg_type}_{gesture}.csv", index = False)



    df_row = {'model_name' : model_ID,
                "emg_type" : emg_type,
                "gesture" : gesture,
    'dim' : dim, 
    'batch_size' : batch_size, 
    'user' : user, 
    'iterations' : iterations,
    "regressor" : reg_type, 
    "ch1" : channels_accuracy_list[0],
    "ch2" : channels_accuracy_list[1],
    "ch3" : channels_accuracy_list[2],
    "ch4" : channels_accuracy_list[3],
    "ch5" : channels_accuracy_list[4],
    "ch6" : channels_accuracy_list[5],
    "ch7" : channels_accuracy_list[6],
    "ch8" : channels_accuracy_list[7],
    "ch9" : channels_accuracy_list[8],
    "ch10" : channels_accuracy_list[9],
    "ch11" : channels_accuracy_list[10],
    "ch12" : channels_accuracy_list[11],
    "ch13" : channels_accuracy_list[12],
    "ch14" : channels_accuracy_list[13],
    "ch15" : channels_accuracy_list[14],
    "ch16" : channels_accuracy_list[15],
    "ch17" : channels_accuracy_list[16],
    "ch18" : channels_accuracy_list[17]
    }












    # accuracy_scores = []

    # reg.fit(embedding_tr, glove_tr_concat[:, 9]) # this is only 9th glove channel - needs to change !! !

    # for i, gesture in enumerate(gestures):

    #     start, end = auxf.cutStimTransition(restim_test, gesture)
        
    #     reg_pred = reg.predict(embedding_test[start:end, :]) #9th glove channel for now - this needs to change for all of them  !!! !

    #     accuracy_score = reg.score(embedding_test[start:end, :], glove_test[:, 9])

    #     accuracy_scores.append(accuracy_score)


    # df_row = {'model_name' : model_ID,
    #           "emg_type" : emg_type,
    # 'dim' : dim, 
    # 'batch_size' : batch_size, 
    # 'user' : user, 
    # 'iterations' : iterations,
    # "regressor" : reg_type, 
    # "r_sq_1_2" : accuracy_scores[0],
    # "r_sq_2_3" : accuracy_scores[1],
    # "r_sq_3_4" : accuracy_scores[2],
    # "r_sq_4_5" : accuracy_scores[3],
    # "r_sq_5_6" : accuracy_scores[4],
    # "r_sq_6_7" : accuracy_scores[5],
    # "r_sq_7_8" : accuracy_scores[6],
    # "r_sq_8_9" : accuracy_scores[7],
    # }
    

    return df_row



# results_df_regression = pd.DataFrame(columns = ['model_name',
#               'emg_type',
#               'dim', 
#               'batch_size', 
#               'user',
#               'iterations',
#               "regressor", 
#               "r_squared", # eventually add gesture
#               ])


df_headers = [
    'model_name', 'emg_type', 'gesture', 'dim', 'batch_size', 'user', 'iterations', 'regressor',
    'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10',
    'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'ch17', 'ch18'
]

results_df_regression = pd.DataFrame(columns=df_headers)


results_list = []
directory_results_df = f'./results_generation/regression_results'
auxf.ensure_directory_exists(directory_results_df)


results_path = f"{directory_results_df}/Regressors.csv"
results_df_regression.to_csv(results_path)

gestures = [1, 2, 3, 4, 5, 6, 7, 8, 9]


for model in list_models:
    for gesture in gestures:
        
        reg_type = "MLP"
        directory_results_df = f'./results_generation/{reg_type}/regression_results'
        auxf.ensure_directory_exists(directory_results_df)

        df_row = runRegression(model_path=model, reg_type=reg_type, gesture = gesture, results_dir = directory_results_df)
        results_list.append(df_row)
        df_row = pd.DataFrame(data=[df_row]) 

        results_stored = pd.read_csv(results_path)
        results_df = pd.concat([results_stored, df_row])
        results_df.to_csv(results_path, index = False)

        reg_type = "GB"
        directory_results_df = f'./results_generation/{reg_type}/regression_results'
        auxf.ensure_directory_exists(directory_results_df)

        df_row = runRegression(model_path=model, reg_type= reg_type, gesture = gesture, results_dir = directory_results_df)
        results_list.append(df_row)
        df_row = pd.DataFrame(data=[df_row]) 

        results_stored = pd.read_csv(results_path)
        results_df = pd.concat([results_stored, df_row])
        results_df.to_csv(results_path, index = False)


