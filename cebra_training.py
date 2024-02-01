
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


# get emg, glove and restimulus data for different users
emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = dtp.getdata(user = 1, dataset = 1)
#emg_u2_d1, glove_u2_d1, restimulus_u2_d1 = dtp.getdata(user = 2, dataset = 1)

# define required channels for the glove data
# zhangcontinuous uses 1, 4, 6, 8, 11, 14 corresponding to [0, 3, 5, 7, 10, 13] with zero index
channels_list = [0, 3, 5, 7, 10, 13]

# use auxf function to extract only the required channels from the glove dataset
glove_u1_d1 = auxf.getGloveChannels(list_channels = channels_list, glove_dataset=glove_u1_d1)
#glove_u2_d1 = auxf.getGloveChannels(list_channels= channels_list, glove_dataset=glove_u2_d1)


# separate training and testing set
#####emg_u1_d1_train, emg_u1_d1_test, glove_u1_d1_train, glove_u1_d1_test = train_test_split(emg_u1_d1, glove_u1_d1, train_size=0.01, random_state=4)
#emg_u2_d1_train, emg_u2_d1_test, glove_u2_d1_train, glove_u2_d1_test = train_test_split(emg_u2_d1, glove_u2_d1, train_size=0.7, random_state=4)


# @cebra.models.register("my-model") # --> add that line to register the model!
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

# # Access the model
# print(cebra.models.get_options('my-model'))

# # define cebra model
# cebra_model1 = CEBRA(
#     model_architecture = "offset10-model",
#     batch_size = (2)**6,
#     temperature_mode='auto',
#     #temperature= 0.9,
#     learning_rate = 0.001,
#     max_iterations = 10000,
#     time_offsets = 5,
#     output_dimension = 6,
#     device = "cuda_if_available",
#     verbose = True,
#     conditional='time',
#     distance = 'cosine',
# )

# cebra_model1 = CEBRA(
#     model_architecture = "offset10-model",
#     batch_size = (2)**4,
#     temperature_mode='auto',
#     #temperature= 0.9,
#     learning_rate = 0.0001,
#     max_iterations = 10000,
#     time_offsets = 1,
#     output_dimension = 6,
#     device = "cuda_if_available",
#     verbose = True,
#     conditional='time',
#     distance = 'cosine',
# )

# define cebra model
cebra_model1 = CEBRA(
    model_architecture = "offset10-model",
    batch_size = (2)**6,
    temperature_mode='auto',
    #temperature= 0.9,
    learning_rate = 0.01,
    max_iterations = 5000,
    time_offsets = 10,
    output_dimension = 8,
    device = "cuda_if_available",
    verbose = True,
    conditional='time',
    distance = 'cosine',
)

# ^ this is the code I used for the plot in the thesis

# cebra_model1 = CEBRA (model_architecture = 'my-model', batch_size=300, max_iterations=1000,
#                  distance = 'cosine', num_hidden_units = 128, conditional='time_delta', output_dimension = 128,
#                 verbose = True, device = 'cuda_if_available', temperature = 1, learning_rate = 3e-4)

# time offsets 5 gave a wierd answer

start_index, end_index =  auxf.cutStimulus(stimulus_u1_d1, 1)

#cebra_model1.fit(emg_u1_d1[start_index:end_index, :])

# cebra_model1.fit(emg_u1_d1)
#cebra_model1.fit(glove_u1_d1[:, :])
cebra_model1.fit(glove_u1_d1[start_index:end_index*8, :])


                #  glove_u1_d1[start_index:(end_index), 0], 
                #  glove_u1_d1[start_index:(end_index), 1], 
                #  glove_u1_d1[start_index:(end_index), 2], 
                #  glove_u1_d1[start_index:(end_index), 3], 
                #  glove_u1_d1[start_index:(end_index), 4], 
                #  glove_u1_d1[start_index:(end_index), 5],
                #  stimulus_u1_d1[start_index:(end_index), :])


# find the training and testing size


# # train cebra model on user 1
#cebra_model1.fit(emg_u1_d1_train, glove_u1_d1_train)

model_name = "./testsavedmodel_2.pt" # time offsets 5 iterations 15000 batch size 300
model_name3 = "./testsavedmodel_3.pt" # time offsets 5 iterations 15000 batch size 300



# send model to cpu and save
#cebra_model1.to('cpu')
cebra_model1.save(model_name3)

#loaded_cebra_model = cebra.CEBRA.load(model_name3)


# generate embedding on the testing data
#embedding = cebra_model1.transform(emg_u1_d1_test)

# embedding = cebra_model1.transform(emg_u1_d1[training_size:training_size+testing_size, :])
# embedding = cebra_model1.transform(emg_u1_d1[:10000, :])
#embedding = cebra_model1.transform(glove_u1_d1[:100000, :])

#embedding = cebra_model1.transform(glove_u1_d1[:100000, :])
embedding = cebra_model1.transform(glove_u1_d1[(end_index*8):(end_index*10), :])




print("embedding shape", embedding.shape)

#accracy = auxf.cKNN(cebra_model1, emg_u1_d1_train, emg_u1_d1_test, glove_u1_d1_train, glove_u1_d1_test)
#print(accracy)

auxf.plotEmbedding(cebra_model1, embedding)


# models = []
# for i in range(12):
#     models.append(deepcopy(single_cebra_model))


# hypothesis -> everything is gone to shit because train_test_split is taking random
# subsets of the data instead of maintaining the temporal aspect!! (alternatively I did not split the rows of the glove data)jowe

# Ok i do not know what to do about the temporal aspect and the training_testing split, I suspect I may
# have to split by trials.? however, I put my glove data back in as separate rows of data input and I got back again my
# sliced shape

# trying again with my simple :40000

