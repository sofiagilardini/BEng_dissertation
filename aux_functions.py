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

    win_size = 3
    win_stride = 2

    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        window = x[start_index:end_index]
        windows.append(window)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows)

    return windows_array

def slidingWindowGlove(x, win_size, win_stride):

    win_size = 3
    win_stride = 2

    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        window = x[start_index:end_index]
        window = np.mean(window)
        windows.append(window)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows)

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
        window_LV = np.log10(np.var(window)) # ? divide by win_size ? 
        windows_LV_list.append(window_LV)
    windows_LV_array = np.array(windows_LV_list)
    print(windows_LV_array.shape)

    return windows_LV_array

def slidingWindowParameters(frequency, size_secs, stride_secs):
    # NB: size and stride should be in Seconds (150ms = 150 * 10**(-3))
    window_size = frequency * size_secs
    window_stride = frequency * stride_secs

    print(window_size, window_stride)

    return window_size, window_stride

## --------- END SLIDING WINDOW FUNCTIONS ---------------- #



def getGloveChannels(list_channels, glove_dataset):
    # this should take as an input a list of channels e.g [0, 1, 4, 5] and give me an np array with only those channels
    # glove dataset is an array of (2million, 18)

    required_channels_data = glove_dataset[:, list_channels]

    return required_channels_data


def plotEmbedding(cebra_model, pre_embedding):

    # plots embedding, loss and temperature

    import matplotlib.pyplot as plt



def plotEmbedding(cebra_model, pre_embedding):
    # Create a figure with a specific size
    fig = plt.figure(figsize=(20, 15))  # Adjust the size as needed

    # Create subplots
    ax1 = fig.add_subplot(221, projection="3d")  # 3D plot for the first embedding
    #ax2 = fig.add_subplot(223, projection="3d")  # 3D plot for the second embedding
    ax3 = fig.add_subplot(222)                   # 2D plot for loss
    ax4 = fig.add_subplot(224)                   # 2D plot for temperature

    # Set the background color for 3D plots to black
    ax1.set_facecolor('black')
    #ax2.set_facecolor('black')

    # First embedding plot
    ax1 = cebra.plot_embedding(pre_embedding, cmap='magma', embedding_labels='time', idx_order=(0, 1, 2), title="Latents: (1,2,3)", ax=ax1, markersize=5, alpha=0.5)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.tick_params(axis='z', colors='white')
    ax1.set_title("Latents (1, 2, 3)", color = 'white', fontsize = 9)
    # # Second embedding plot
    # ax2 = cebra.plot_embedding(pre_embedding, cmap='magma', embedding_labels='time', idx_order=(3, 4, 5), title="Latents: (4,5,6)", ax=ax2, markersize=5, alpha=0.5)
    # ax2.tick_params(axis='x', colors='white')
    # ax2.tick_params(axis='y', colors='white')
    # ax2.tick_params(axis='z', colors='white')
    # ax2.set_title("Latents (3, 4, 5)", color = 'white', fontsize = 9)



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

