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


directory_loadmodels = "./results_generation/model_training/behaviour_contrastive"

list_models = []

# load all regression models
for dirpath, dirnames, filenames in os.walk(directory_loadmodels):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        
        if file_path.__contains__(".pt"):
            if file_path.__contains__("mintemp1_"): # optimal minimum temperature for regression
                model_path = file_path
                list_models.append(model_path)
                print(model_path)


def plotRegressionResults(model_path):
    
    type_training, user, emg_type, batch_size, min_temp, iterations = auxf.extract_model_params(model_path)

    if emg_type == "raw":
        rawBool = True

    else: 
        rawBool = False


    restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    restim_test = auxf.getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)


    glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, mode='glove', rawBool = rawBool)
    glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, mode='glove', rawBool = rawBool)
    glove_test = auxf.getProcessedData(user = user, dataset = 3, mode='glove', rawBool = rawBool)


    gesture_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    glove_channel_list = np.arange(0, 18, 1)

    # 1. I need the real glove data. i.e I need the restim data to cut to the glove data 

    # focus on GB for now

    reg_type_list = ["MLP", "GB"] # need to add PSLR
    
    reg_colour_list = ["red", "magenta"]

    for gesture in gesture_list:



        for index, glove_channel in enumerate(glove_channel_list):

            fig, ax = plt.subplots(figsize = (10, 10))

            start, end = auxf.cutStimTransition(restim_test, required_stimulus=gesture)
            glove_truth = glove_test[start:end, glove_channel]

            xvals = np.arange(0, len(glove_truth), 1)
            ax.plot(xvals, glove_truth, label = 'Ground Truth', color = 'black')

            for index_reg, reg_type in enumerate(reg_type_list):
                
                gesture_prediction_path = f"./results_generation/regression_results/regression_predictions/{reg_type}/User{user}/Gesture{gesture}"

                for dirpath, dirname, filenames in os.walk(gesture_prediction_path):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            reg_pred_df = pd.read_csv(file_path)

                            reg_pred_channel = reg_pred_df.iloc[index]
                            reg_pred_channel = auxf.lowpass_filter(data = reg_pred_channel, cutoff=200, order = 4, fs= 2000)
                            ax.plot(xvals, reg_pred_channel, label = f"{reg_type}", color = reg_colour_list[index_reg])
                        
            ax.legend()
            ax.set_xlabel("Windows")
            ax.set_ylabel("Joint state")  # TODO: change units

            plot_path = f"./results_generation/regression_results/regression_predictions/trajectory_plots/Gesture{gesture}/Channel{glove_channel}"
            auxf.ensure_directory_exists(plot_path)

            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User: {user}, Channel: {glove_channel}")
            plt.savefig(f"{plot_path}/User{user}_Gesture{gesture}_Channel{glove_channel}.png")
            #plt.show()

                







for model_path in list_models: 
    if model_path.__contains__("user_10"):
        plotRegressionResults(model_path=model_path)


