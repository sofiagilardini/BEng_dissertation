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
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats 


plt.rcParams.update(
    {
        "lines.markersize": 20,  # Big points
        "font.size": 15,  # Larger font
        "xtick.major.size": 10.0,  # Bigger xticks
        "ytick.major.size": 10.0,  # Bigger yticks
    }
)


"""

This script contains all the necessary auxiliary functions used in this work - both in preliminary
tests and tests that were included in the final results. 

They are called in all scripts in this work by auxf.()


"""



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




def getProcessedData(user: int, dataset: int, mode: str, rawBool: bool):

    """
    mode = ["glove", "restimulus"]
    
    """

    if mode == 'emg':
        raise ValueError("Incorrect mode 'emg'. Use getProcessedEMG()")


    winsize = 128
    winstride = 52

    if rawBool: 
        data = np.load(f"./processed_data/{mode}_raw/dataset{dataset}/{mode}_{user}_{dataset}.npy")

    if not rawBool: 
        data = np.load(f"./processed_data/{mode}/dataset{dataset}/{mode}_{user}_{dataset}_{winsize}_{winstride}.npy")

    return data

def getMappedGlove(user: int, dataset: int):

    """
    get the glove data that has been linearly transformed in ./glove_linear_mapping/linear_mapping_processing.py

    """

    data = np.load(f"./processed_data/glove_mapped/user{user}_dataset{dataset}.npy")

    return data


def getProcessedEMG(user: int, dataset: int, type_data: str):

    """
    type_data : ["raw", "all", "RMS"]
    
    """

    if type_data == "raw":
        rawBool = True
    
    elif type_data !=  "raw":
        rawBool = False


    winsize = 128
    winstride = 52
    cutoff = 450

    if rawBool:
        data = np.load(f"./processed_data/emg_{type_data}/dataset{dataset}/emg_{user}_{dataset}.npy")

    
    if not rawBool:
        data = np.load(f"./processed_data/emg_{type_data}/dataset{dataset}/emg_{user}_{dataset}_{cutoff}_{winsize}_{winstride}_{type_data}.npy")


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
        window = x[start_index:end_index]


        # Calculate the mode of the window
        mode_result = stats.mode(window)
        
        if np.isscalar(mode_result.mode):
            window_mode = mode_result.mode
        else:
            window_mode = mode_result.mode[0]
        
        windows.append(window_mode)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows) # this returns an array where each element is the
    # final data point for that window 

    return windows_array



def WL(windows_array, win_size, win_stride):

    windows_WL_list = []

    for window in windows_array:
        window_WL = 1/win_size * np.sum(np.abs(np.diff(window)))
        windows_WL_list.append(window_WL)
    
    windows_WL_array = np.array(windows_WL_list) # should be one dimensional (num_windows,)

    return windows_WL_array

def LV(windows_array, win_size):
    windows_LV_list = []
    for window in windows_array:

        # log smoothing needed 
        epsilon = 1e-15

        window_LV = np.log10(np.var(window)+epsilon) # 
        windows_LV_list.append(window_LV)
    windows_LV_array = np.array(windows_LV_list)

    return windows_LV_array


def RMS(windows_array, win_size, win_stride):

    windows_RMS_list = []

    for window in windows_array:

        window_RMS = np.sqrt(np.mean(np.square(window)))
        windows_RMS_list.append(window_RMS) 
    windows_RMS_array = np.array(windows_RMS_list)

    return windows_RMS_array


# SSC not used in the end

def SSC(windows_array, win_size, win_stride, thr= 0):

    windows_SSC_list = []

    for window in windows_array:

        diff1 = window[1:-1] - window[: -2]
        diff2 = window[1-1] - window[2:]

        ssc = (diff1 * diff2) > thr
        
        ssc_count = np.sum(ssc)

        windows_SSC_list.append(ssc_count)
    
    windows_SSC_array = np.array(windows_SSC_list)

    return windows_SSC_array



def slidingWindowParameters(frequency, size_secs, stride_secs):
    # NB: size and stride should be in Seconds (150ms = 150 * 10**(-3))
    window_size = frequency * size_secs
    window_stride = frequency * stride_secs

    return window_size, window_stride

## --------- END SLIDING WINDOW FUNCTIONS ---------------- #


### ------- EMG , GLOVE AND RESTIMULUS PROCESSING START ----- ###

def emg_process(cutoff_val, size_val, stride_val, user, dataset, order, feat_ID: str):
        

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    if feat_ID == 'raw':
        raw_bool = True
        WL_bool = False
        LV_bool = False
        RMS_bool = False
        # SSC_bool = False 
    
    if feat_ID == 'all':
        raw_bool = False
        WL_bool = True
        LV_bool = True
        RMS_bool = True
        # SSC_bool = True
    
    if feat_ID == 'RMS':
        raw_bool = False
        WL_bool = False
        LV_bool = False
        RMS_bool = True
        # SSC_bool = False

    
    fs = 2000

    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

    size = int(size)
    stride = int(stride)
    # 2. get the data: input is user and which dataset 

    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

    # 3. low-pass filter the EMG data and extract features

    for i in range(emg_u1_d1.shape[1]):
        emg_u1_d1[:, i] = lowpass_filter(emg_u1_d1[:, i], cutoff_val, fs, order)

    del glove_u1_d1, stimulus_u1_d1

    if raw_bool: 


        # standardise the data 

        scaler = StandardScaler()
        emg_u1_d1 = scaler.fit_transform(emg_u1_d1)

        emg_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/emg_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/emg_{emg_data_ID}.npy", emg_u1_d1)


    if not raw_bool:

        emg_windows = []

        # Loop through each channelc
        num_channelsEMG = emg_u1_d1.shape[1] 


        # low pass and features -> 

        for i in range(num_channelsEMG):
            
            
            emg_window = slidingWindowEMG(emg_u1_d1[:, i], size, stride)

            if WL_bool: 
                emg_window_WL = WL(emg_window, size, stride) 
                emg_windows.append(emg_window_WL)



            if LV_bool:
                emg_window_LV = LV(emg_window, size)
                emg_windows.append(emg_window_LV)



            if RMS_bool:
                emg_window_RMS = RMS(emg_window, size, stride)
                emg_windows.append(emg_window_RMS)



            # emg_window_SSC = SSC(emg_window, size, stride)
            # SSC_bool = True
            # print("SSC", emg_window_SSC)

            # emg_windows.append(emg_window_SSC)


        emg_windows_stacked = np.array(emg_windows)
        emg_windows_stacked = np.transpose(emg_windows_stacked)

        scaler = StandardScaler()
        emg_windows_stacked = scaler.fit_transform(emg_windows_stacked)


        directory = f'./processed_data/emg_{feat_ID}/dataset{dataset}'

        ensure_directory_exists(directory)


        emg_data_ID = f"{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}"

        
        np.save(f"{directory}/emg_{emg_data_ID}.npy", emg_windows_stacked)



    return emg_data_ID



def glove_process(size_val, stride_val, user, dataset, rawBool: bool):

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, stimulus_u1_d1

    if rawBool == True:  

        scaler = StandardScaler()
        glove_u1_d1 = scaler.fit_transform(glove_u1_d1)

        glove_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/glove_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/glove_{glove_data_ID}.npy", glove_u1_d1)



    if rawBool == False:
    
        size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        size = int(size)
        stride = int(stride)
        

        # perform sliding windows on the data
            
        glove_data = []

        for i in range(glove_u1_d1.shape[1]):

            glove_data.append(slidingWindowGlove(glove_u1_d1[:, i], size, stride))

        
        glove_data_array = np.array(glove_data).T

        scaler = StandardScaler()
        glove_data_array = scaler.fit_transform(glove_data_array)


        directory = f'./processed_data/glove/dataset{dataset}'

        def ensure_directory_exists(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)


        ensure_directory_exists(directory)


        glove_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

        
        np.save(f"{directory}/glove_{glove_data_ID}.npy", glove_data_array)




    return glove_data_ID




def restimulusProcess(size_val, stride_val, user, dataset, rawBool: bool):


    emg_u1_d1, glove_u1_d1, restimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, glove_u1_d1

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    if rawBool == True:     

        restimulus_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/restimulus_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/restimulus_{restimulus_data_ID}.npy", restimulus_u1_d1.astype(int))


    if rawBool == False:
    
        size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        size = int(size)
        stride = int(stride)
        

        restimulus_data = []

        for i in range(restimulus_u1_d1.shape[1]):
            restimulus_data.append(slidingWindowRestimulus(restimulus_u1_d1[:, i], size, stride))

        restimulus_data_array = np.array(restimulus_data).T

        directory = f'./processed_data/restimulus/dataset{dataset}'

        def ensure_directory_exists(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)


        ensure_directory_exists(directory)


        restimulus_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

        
        np.save(f"{directory}/restimulus_{restimulus_data_ID}.npy", restimulus_data_array.astype(int))



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

    final_index = len(stimulus_data)

    start_index_flag = False
    start_index = 0
    end_index = 0

    for i in range(len(stimulus_data_list)):
        if not start_index_flag and stimulus_data_list[i] == [required_stimulus]:
            start_index = i
            start_index_flag = True
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
    
        if start_index_flag and stimulus_data_list[i] == [0]:
            start_break_flag = True

        if start_index_flag and start_break_flag and stimulus_data_list[i] == [0]:
            print("in rest index at ", i)
        
        if start_index_flag and start_break_flag and stimulus_data_list[i] != [0]:
            end_index = i - 1
            break



    return start_index, end_index


def cutStimTransition(stimulus_data, required_stimulus):

    """
    This function is for the TEST dataset (dataset 3). It returns the indices for transition from the 
    gesture i (required stimulus) to i + 1
    
    """



    stimulus_data_list = stimulus_data.tolist()

    start_index = None
    end_index = None
    in_transition = False
    rep_counter = 0
    in_rep = False
    in_next = False


    # Iterate over the stimulus data
    for i in range(len(stimulus_data_list) - 1):


        if not in_rep and stimulus_data_list[i] == [required_stimulus]:
            rep_counter += 1
            in_rep = True
        
        if in_rep and stimulus_data_list[i] == [0]:
            in_rep = False

        if rep_counter == 1:
            start_index = i+2
    
        
        if stimulus_data_list[i] == [required_stimulus + 1]:

            in_next = True

        if in_next and stimulus_data_list[i] == [0]:
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

    heatmap_array = np.zeros((1, len(k_list)))

    models_name_list = []

    for indx_k, k in enumerate(k_list): 


        train_embedding = model.transform(emg_train)
        test_embedding = model.transform(emg_test)


        KNN_decoder = cebra.KNNDecoder(n_neighbors= k, metric = 'cosine')

        KNN_decoder.fit(train_embedding, rest_train)


        KNN_pred = KNN_decoder.predict(test_embedding)

        KNN_decoder = cebra.KNNDecoder(n_neighbors= k, metric = 'cosine')

        # KNN_decoder.fit(emg_train, rest_train)

        KNN_decoder.dit(train_embedding, rest_train)

        KNN_pred = KNN_decoder.predict(emg_test)

        accuracy = accuracy_score(rest_test, KNN_pred)

        print(f'accuracy {indx_k}', accuracy)

        rounded_accuracy = round(accuracy, 3)


        # row model, column k
        heatmap_array[0, indx_k] = rounded_accuracy

        # row ; model
        # column ; k

        heatmap_df = pd.DataFrame(data = heatmap_array)
        heatmap_df['user'] = user

    
        print(heatmap_array)


        
    return heatmap_df




def PCA_KNN_heatmap_df(emg_train, emg_test, rest_train, rest_test, k_list: list, user: int):


    heatmap_array = np.zeros((1, len(k_list)))

    models_name_list = []

    for indx_k, k in enumerate(k_list): 

        pca = PCA(n_components=6)

        train_embedding = pca.fit_transform(emg_train)

        test_embedding = pca.transform(emg_test)

        KNN_decoder = cebra.KNNDecoder(n_neighbors= k, metric = 'cosine')

        KNN_decoder.fit(train_embedding, rest_train)

        KNN_pred = KNN_decoder.predict(test_embedding)

        accuracy = accuracy_score(rest_test, KNN_pred)

        print(f'accuracy {indx_k}', accuracy)

        rounded_accuracy = round(accuracy, 3)


        # row model, column k
        heatmap_array[0, indx_k] = rounded_accuracy

        # row ; model
        # column ; k

        heatmap_df = pd.DataFrame(data = heatmap_array)
        heatmap_df['user'] = user
        print("inside PCA for user", user)


    
        print(heatmap_array)
    
    return heatmap_df





def runCebraCheck(dataset):

    """
    first order check that all data runs: 
    """

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

    model.fit(dataset)

    return model



def ensure_directory_exists(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)



def extract_model_params(path):
    
    """"

    returns type_training, user, emg_type, batch_size, mintemp, iterations
    
    """

    list_dirs = path.split("/")
    
    type_training = list_dirs[4]

    filename = os.path.basename(path)

    parts = filename.split('_')
    
    user = parts[1]
    emg_type = parts[2]
    batch_size = parts[4]
    mintemp = parts[5].replace('mintemp', '')
    iterations = parts[6].split('.')[0].replace('it', '')

    return type_training, user, emg_type, batch_size, mintemp, iterations
    

def plotRegressionResults(model_path):
    
    type_training, user, emg_type, batch_size, min_temp, iterations = extract_model_params(model_path)

    if emg_type == "raw":
        rawBool = True

    else: 
        rawBool = False


    restim_tr1 = getProcessedData(user = user, dataset = 1, mode='restimulus', rawBool = rawBool)
    restim_tr2 = getProcessedData(user = user, dataset = 2, mode='restimulus', rawBool = rawBool)
    restim_test = getProcessedData(user = user, dataset = 3, mode='restimulus', rawBool = rawBool)


    glove_tr1 = getProcessedData(user = user, dataset = 1, mode='glove', rawBool = rawBool)
    glove_tr2 = getProcessedData(user = user, dataset = 2, mode='glove', rawBool = rawBool)
    glove_test = getProcessedData(user = user, dataset = 3, mode='glove', rawBool = rawBool)


    gesture_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    glove_channel_list = np.arange(1, 19, 1)

    # 1. I need the real glove data. i.e I need the restim data to cut to the glove data 

    # focus on GB for now

    reg_type_list = ["MLP", "GB"] # need to add PSLR
    
    reg_colour_list = ["red", "magenta"]

    for gesture in gesture_list:



        for index, glove_channel in enumerate(glove_channel_list):

            fig, ax = plt.subplots(figsize = (10, 10))

            start, end = cutStimTransition(restim_test, required_stimulus=gesture)
            glove_truth = glove_test[start:end, index]

            xvals = np.arange(0, len(glove_truth), 1)
            ax.plot(xvals, glove_truth, label = 'Ground Truth', color = 'black')

            for index_reg, reg_type in enumerate(reg_type_list):
                
                gesture_prediction_path = f"./results_generation/regression_results/regression_predictions/{reg_type}/User{user}/Gesture{gesture}"

                for dirpath, dirname, filenames in os.walk(gesture_prediction_path):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            reg_pred_df = pd.read_csv(file_path)

                            reg_pred_channel = reg_pred_df.iloc[index]
                            ax.plot(xvals, reg_pred_channel, label = f"{reg_type}", color = reg_colour_list[index_reg])
                        
            ax.legend()
            ax.set_xlabel("Windows")
            ax.set_ylabel("Joint state")  # TODO: change units

            plot_path = f"./results_generation/regression_results/regression_predictions/trajectory_plots/Gesture{gesture}/Channel{glove_channel}"
            ensure_directory_exists(plot_path)

            ax.set_title(f"Gesture {gesture} to {gesture + 1}, User: {user}, Channel: {glove_channel}")
            plt.savefig(f"{plot_path}/User{user}_Gesture{gesture}_Channel{glove_channel}.png")
            #plt.show()

    return plot_path
                

