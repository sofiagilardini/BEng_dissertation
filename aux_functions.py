

# -------   PREAMBLE ------------- #

import matplotlib.pyplot as plt
import scipy.io 
import numpy as np 
import cebra
from cebra import CEBRA
import cebra.models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd


plt.rcParams.update(
    {
        "lines.markersize": 20,  # Big points
        "font.size": 15,  # Larger font
        "xtick.major.size": 10.0,  # Bigger xticks
        "ytick.major.size": 10.0,  # Bigger yticks
    }
)

# -------   PREAMBLE ------------- #



# ---------- DATA PIPELINE ------------------- #

# user is S1-S12 and dataset is A1-A3 (note A3 only has 2 repetitions of each movement)
def getdata(user, dataset):
    data = scipy.io.loadmat(f"./datasets/S{user}_E1_A{dataset}.mat")
    for key in data:
        print(key)
    emg_data = data['emg']
    glove_data = data['glove']
    restimulus_data = data['restimulus']

    return emg_data, glove_data, restimulus_data



def getProcessedData(user: int, dataset: int, type_data: str, mode: str):

    winsize = 128
    winstride = 52
    cutoff = 450
    feat_ID = '0102'

    if mode == 'emg':

    # data = scipy.io.loadmat(f"./processed_data/{type_data}_data/{mode}_data_processed/{mode}_{user}_{dataset}_{cutoff}_{winsize}_{winstride}_{feat_ID}")
        data = np.load(f"./{type_data}_data/{mode}_data_processed/{mode}_{user}_{dataset}_{cutoff}_{winsize}_{winstride}_{feat_ID}.npy")

    else: 
        data = np.load(f"./{type_data}_data/{mode}_data_processed/{mode}_{user}_{dataset}_{winsize}_{winstride}.npy")


    return data





# ----------- FILTERING ------------------------------#

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --------- FILTERING END -------------------------#



## --------- SLIDING WINDOW FUNCTIONS ---------------- #

def slidingWindowEMG(x, win_size, win_stride):


    # #num_windows = int(1 + (len(x) - win_size) // win_stride)

    # # Calculate the number of full windows that can be formed
    # num_windows = int((len(x) - win_size) / win_stride) + 1

    # # Ensure we don't create a window that goes beyond the length of x
    # num_windows = min(num_windows, (len(x) - win_size) // win_stride + 1)


    # # Calculate the maximum length that fits the window size and stride
    # max_length = win_size + ((len(x) - win_size) // win_stride) * win_stride

    # print('length of x was', len(x))
    # # Truncate x to the maximum length
    # x_truncated = x[:max_length]
    # print('length x_truc', len(x_truncated))

    # Perform the sliding window operation
    num_windows = (len(x) - win_size) // win_stride + 1


    windows = []
    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        # Check if the end index goes beyond the array length
        if end_index > len(x):
            break  # Break the loop if the window exceeds the array length
        window = x[start_index:end_index]
        windows.append(window)

    # Convert the list of windows to a numpy array
    windows_array = np.array(windows)

    return windows_array # this returns an array where each element is a window of data -> for feature extraction




def slidingWindowGlove(x, win_size, win_stride):


    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):


        start_index = i * win_stride
        end_index = start_index + win_size
        window = x[start_index:end_index]
        # window = np.mean(window) # here I am doing mean -> I want to be doing EWMA
        # windows.append(window)

        # not doing EWMA but taking mean of last 40% of the window

        # Calculate start index for the last 40% of the window
        mean_start_index = start_index + int(win_size * 0.6)


        # Take only the last 40% of the window
        end_segment_window = x[mean_start_index:end_index]

        # Calculate the mean of the last 40%
        window_mean = np.mean(end_segment_window)

        windows.append(window_mean)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows) # this returns an array where each element is the
    # final data point for that window 

    return windows_array


def slidingWindowRestimulus(x, win_size, win_stride):


    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        #window = x[start_index:end_index]
        #print("window", window)

        # window = np.mean(window) # here I am doing mean -> I want to be doing EWMA
        # windows.append(window)

        #not doing EWMA but taking mean of last 40% of the window

        # # Calculate start index for the last 40% of the window
        mean_start_index = start_index + int(win_size * 0.6)

        # print("mean start index", mean_start_index)

        # Take only the last 40% of the window
        end_segment_window = x[mean_start_index:end_index]

        # print(end_segment_window)        

        # Calculate the mean of the last 40%
        window_mean = np.mean(end_segment_window)
        #window_mean = np.mean(window)


        windows.append(window_mean)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows) # this returns an array where each element is the
    # final data point for that window 

    return windows_array



def WL(windows_array, win_size, win_stride):

    windows_WL_list = []

    for window in windows_array:
        window_WL = 1/win_size * np.sum(np.abs(np.diff(window)))
        print('window_WL', window_WL)
        windows_WL_list.append(window_WL)
    
    windows_WL_array = np.array(windows_WL_list) # should be one dimensional (num_windows,)

    return windows_WL_array

def LV(windows_array, win_size):
    windows_LV_list = []
    for window in windows_array:
        print('var window', np.var(window))

        # log smoothing needed 

        epsilon = 1e-15

        window_LV = np.log10(np.var(window)+epsilon) # 
        print('window LV', window_LV)
        windows_LV_list.append(window_LV)
    windows_LV_array = np.array(windows_LV_list)
    #print(windows_LV_array.shape)

    return windows_LV_array

def slidingWindowParameters(frequency, size_secs, stride_secs):
    # NB: size and stride should be in Seconds (150ms = 150 * 10**(-3))
    window_size = frequency * size_secs
    window_stride = frequency * stride_secs

    #print(window_size, window_stride)

    return window_size, window_stride

## --------- END SLIDING WINDOW FUNCTIONS ---------------- #


### ------- EMG , GLOVE AND RESTIMULUS PROCESSING START ----- ###

def emg_process(cutoff_val, size_val, stride_val, user, dataset, order):

        fs = 2000

        size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        size = int(size)
        stride = int(stride)
        # 2. get the data: input is user and which dataset (dataset should only be 1 or 2)

        emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

        # 3. low-pass filter the EMG data and extract features

        for i in range(emg_u1_d1.shape[1]):
            emg_u1_d1[:, i] = lowpass_filter(emg_u1_d1[:, i], cutoff_val, fs, order)


        emg_windows = []

        # Loop through each channelc
        num_channelsEMG = emg_u1_d1.shape[1] 


        # low pass and features = [WL and LV] -> 
        # for i in range(num_channelsEMG):

        for i in range(num_channelsEMG):
            emg_window = slidingWindowEMG(emg_u1_d1[:, i], size, stride)
            print("emg windowing at step ", i)
            emg_window_WL = WL(emg_window, size, stride) 
            #print("emg WL extracting at step ", i)
            WL_bool = True
            emg_window_LV = LV(emg_window, size)
            #print("emg LV extracting at step ", i)
            LV_bool = True
            print('length emg window', len(emg_window))
            emg_windows.append(emg_window_WL)
            emg_windows.append(emg_window_LV)


        emg_windows_stacked = np.array(emg_windows)
        emg_windows_stacked = np.transpose(emg_windows_stacked)

        # if np.isinf(emg_windows_stacked).any() or np.isnan(emg_windows_stacked).any():
        # # Handle the presence of inf or NaN. Options might include:
        # # - Raising an error
        # # - Replacing inf and NaN with a specific value
        # # - Removing rows/columns containing inf or NaN

        # # For example, to raise an error:
        #     raise ValueError("The EMG processed data contains 'inf' or 'NaN' values.")


        if WL_bool == True and LV_bool == True:
            feat_ID = '0102'

        
        if dataset == 1:
            np.save(f"./training_data/emg_data_processed/emg_{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}.npy", emg_windows_stacked)
        
        elif dataset == 2:
            np.save(f"./validation_data/emg_data_processed/emg_{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}.npy", emg_windows_stacked)

        elif dataset == 3:
            np.save(f"./test_data/emg_data_processed/emg_{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}.npy", emg_windows_stacked)



        # np.save(f"./emg_data_processed/emg_{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}.npy", emg_windows_stacked)
        # print("saved EMG data")


        emg_data_ID = f"{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}"

        return emg_data_ID




def glove_process(size_val, stride_val, user, dataset):
     
    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

    size = int(size)
    stride = int(stride)
    
    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, stimulus_u1_d1


    # perform sliding windows on the data
        
    glove_data = []

    for i in range(glove_u1_d1.shape[1]):

        glove_data.append(slidingWindowGlove(glove_u1_d1[:, i], size, stride))

    
    glove_data_array = np.array(glove_data).T


    if dataset == 1:
        np.save(f"./training_data/glove_data_processed/glove_{user}_{dataset}_{size_val}_{stride_val}.npy", glove_data_array)

    elif dataset == 2:
        np.save(f"./validation_data/glove_data_processed/glove_{user}_{dataset}_{size_val}_{stride_val}.npy", glove_data_array)

    elif dataset == 3:
        np.save(f"./test_data/glove_data_processed/glove_{user}_{dataset}_{size_val}_{stride_val}.npy", glove_data_array)



    # np.save(f"./glove_data_processed/glove_{user}_{dataset}_{size_val}_{stride_val}.npy", glove_data_array)
    # print("saved glove data")

    glove_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

    return glove_data_ID




def restimulusProcess(size_val, stride_val, user, dataset):
    
    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

    size = int(size)
    stride = int(stride)
    
    emg_u1_d1, glove_u1_d1, restimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, glove_u1_d1

    restimulus_data = []

    for i in range(restimulus_u1_d1.shape[1]):
        restimulus_data.append(slidingWindowRestimulus(restimulus_u1_d1[:, i], size, stride))

    restimulus_data_array = np.array(restimulus_data).T

    if dataset == 1:
        np.save(f"./training_data/restimulus_data_processed/restimulus_{user}_{dataset}_{size_val}_{stride_val}.npy", restimulus_data_array)

    elif dataset == 2:
        np.save(f"./validation_data/restimulus_data_processed/restimulus_{user}_{dataset}_{size_val}_{stride_val}.npy", restimulus_data_array)

    elif dataset == 3:
        np.save(f"./test_data/restimulus_data_processed/restimulus_{user}_{dataset}_{size_val}_{stride_val}.npy", restimulus_data_array)


    # np.save(f"./restimulus_data_processed/restimulus_{user}_{dataset}_{size_val}_{stride_val}.npy", restimulus_data_array)
    # print("saved restimulus data")

    restimulus_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

    return restimulus_data_ID




### ------- EMG , GLOVE AND RESTIMULUS PROCESSING END ----- ###



def getGloveChannels(list_channels, glove_dataset):
    # this should take as an input a list of channels e.g [0, 1, 4, 5] and give me an np array with only those channels
    # glove dataset is an array of (2million, 18)

    required_channels_data = glove_dataset[:, list_channels]

    return required_channels_data





def plotEmbedding(cebra_model, pre_embedding):
    # Create a figure with a specific size
    fig = plt.figure(figsize=(20, 15))  # Adjust the size as needed

    # Create subplots
    ax1 = fig.add_subplot(221, projection="3d")  # 3D plot for the first embedding
    ax2 = fig.add_subplot(223, projection="3d")  # 3D plot for the second embedding
    ax3 = fig.add_subplot(222)                   # 2D plot for loss
    ax4 = fig.add_subplot(224)                   # 2D plot for temperature

    # Set the background color for 3D plots to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    # First embedding plot
    ax1 = cebra.plot_embedding(pre_embedding, cmap='magma', embedding_labels='time', idx_order=(0, 1, 2), title="Latents: (1,2,3)", ax=ax1, markersize=5, alpha=0.5)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.tick_params(axis='z', colors='white')
    ax1.set_title("Latents (1, 2, 3)", color = 'white', fontsize = 9)
    # Second embedding plot
    ax2 = cebra.plot_embedding(pre_embedding, cmap='magma', embedding_labels='time', idx_order=(3, 4, 5), title="Latents: (4,5,6)", ax=ax2, markersize=5, alpha=0.5)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='z', colors='white')
    ax2.set_title("Latents (3, 4, 5)", color = 'white', fontsize = 9)



    # Loss plot
    cebra.plot_loss(cebra_model, ax=ax3)

    # Temperature plot
    cebra.plot_temperature(cebra_model, ax=ax4)
    plt.subplots_adjust(wspace=1.5, hspace=0.6)

    # ax3.set_title("Loss", fontsize = 9)
    # ax4.set_title("Temperature", fontsize = 9)

    # Adjust layout
    plt.tight_layout()

    # Show the figure with all subplots
    plt.show()

def cKNN(model, neuraldata_tr, neuraldata_test, behaviour_train, behaviour_test):\

    train_embedding = model.transform(neuraldata_tr)
    test_embedding = model.transform(neuraldata_test)

    cKNN_decoder = cebra.KNNDecoder(n_neighbors = 36, metric = 'cosine')

    cKNN_decoder.fit(train_embedding, behaviour_train)

    cKNN_pred = cKNN_decoder.predict(test_embedding)

    accuracy_cKNN = accuracy_score(behaviour_test, cKNN_pred)


    return accuracy_cKNN 


def cutStimulus(stimulus_data, required_stimulus):
    # takes as an input the stimulus data, and returns one numpy array with the desired stimulus
    # what I need is the first index (where i starts) and the last index (where i + 1 starts)


    stimulus_data_list = stimulus_data.tolist()

    start_index_flag = False
    start_index = 5
    end_index = 0

    for i in range(len(stimulus_data_list)):
        if not start_index_flag and stimulus_data_list[i] == [required_stimulus]:
            start_index = i
            start_index_flag = True
            print("start index has changed at i", i)
        if stimulus_data_list[i] == [required_stimulus + 1]:
            end_index = i-1
            break
    
    return start_index, end_index

        

def cutStimulus2(stimulus_data, required_stimulus):
    # takes as an input the stimulus data, and returns one numpy array with the desired stimulus
    # what I need is the first index (where i starts) and the last index (where i + 1 starts)

    # this is a version of the cutStimulus function where I get only one instance of 111111100000 (to plot tragectory)

    stimulus_data_list = stimulus_data.tolist()

    start_index_flag = False
    start_index = 0
    end_index = 0
    start_break_flag = False

    for i in range(len(stimulus_data_list)):
        if not start_index_flag and stimulus_data_list[i] == [required_stimulus]:
            start_index = i
            start_index_flag = True
            print("start index has changed at i (2)", i)
    
        if start_index_flag and stimulus_data_list[i] == [0]:
            start_break_flag = True

        if start_index_flag and start_break_flag and stimulus_data_list[i] == [0]:
            print("in rest index at ", i)
        
        if start_index_flag and start_break_flag and stimulus_data_list[i] != [0]:
            end_index = i - 1
            break



    return start_index, end_index



def KNN_heatmap(model_list: list, emg_train, emg_test, rest_train, rest_test, k_list: list):

    # model list can also just be one model, but for scalability purposes 

    heatmap_array = np.zeros((len(model_list), len(k_list)))

    models_name_list = []

    for indx_model, model_path in enumerate(model_list):
        for indx_k, k in enumerate(k_list): # how to determine k -> relevance to number of features ? 

            model = cebra.CEBRA.load(model_path)

            filename = model_path.split('_')[-1].split('.csv')[0]

            models_name_list.append(filename)

            train_embedding = model.transform(emg_train)
            test_embedding = model.transform(emg_test)


            KNN_decoder = cebra.KNNDecoder(n_neighbors= k, metric = 'cosine')

            KNN_decoder.fit(train_embedding, rest_train)

            KNN_pred = KNN_decoder.predict(test_embedding)

            accuracy = accuracy_score(rest_test, KNN_pred)

            print(f'accuracy {indx_model, indx_k}', accuracy)

            # row model, column k
            heatmap_array[indx_model, indx_k] = accuracy

            # row = model
            # column = k

            print(heatmap_array)



    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_array, vmin = 0, vmax = 1)
    plt.xticks(ticks=np.arange(len(k_list)), labels=k_list)
    # plt.yticks(ticks = np.arange(len(models_name_list)), labels = models_name_list)
    # plt.yticks(rotation = 90)
    plt.title("Heatmap for model and k parameters")
    plt.xlabel("K")  # how will this label the models ? 
    plt.ylabel("Model")
    plt.show()



def KNN_heatmap_df(model, emg_train, emg_test, rest_train, rest_test, k_list: list, user: int):

    # model list can also just be one model, but for scalability purposes 

    heatmap_array = np.zeros((1, len(k_list)))

    models_name_list = []

    for indx_k, k in enumerate(k_list): # how to determine k -> relevance to number of features ? 

        #model = cebra.CEBRA.load(model_path)

        #filename = model_path.split('_')[-1].split('.csv')[0]

        #models_name_list.append(filename)

        train_embedding = model.transform(emg_train)
        test_embedding = model.transform(emg_test)


        KNN_decoder = cebra.KNNDecoder(n_neighbors= k, metric = 'cosine')

        KNN_decoder.fit(train_embedding, rest_train)

        KNN_pred = KNN_decoder.predict(test_embedding)

        accuracy = accuracy_score(rest_test, KNN_pred)

        print(f'accuracy {indx_k}', accuracy)

        rounded_accuracy = round(accuracy, 3)


        # row model, column k
        heatmap_array[0, indx_k] = rounded_accuracy

        # row = model
        # column = k

        #heatmap_df = pd.DataFrame(data = heatmap_array, index = f"User: {user}")
        heatmap_df = pd.DataFrame(data = heatmap_array)

    
        print(heatmap_array)



    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_array, vmin = 0, vmax = 1)
    plt.xticks(ticks=np.arange(len(k_list)), labels=k_list)
    # plt.yticks(ticks = np.arange(len(models_name_list)), labels = models_name_list)
    # plt.yticks(rotation = 90)
    plt.title("Heatmap for model and k parameters")
    plt.xlabel("K")
    plt.savefig(f"./classification_results/KNN_heatmap_user{user}.png")
    #plt.show()

    return heatmap_df