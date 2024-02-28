

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
import seaborn as sns
import os

from sklearn.decomposition import PCA



models_list = ["offset10-model", 
                'offset5-model',
                "offset36-model",
                "offset36-model-dropout",
                "offset36-model-more-dropout"]



# # the subsample ones make the dataset smaller

# # models_list = [
# #                 "offset40-model-4x-subsample",
# #                 "offset20-model-4x-subsample", 
# #                 "offset4-model-2x-subsample", 
# #                 "supervised10-model",
# #                 "offset36-model",
# #                 "offset36-model-dropout",
# #                 "offset36-model-more-dropout"]

# # supervised 10 model: ValueError: It is not yet supported to run non-convolutional models with receptive fields/offsets larger than 1 via the sklearn API. Please use a different model, or revert to the pytorch API for training.



time_offsets_list = [10, 20, 50]
dm_list = [6, 9, 12]
batch_size_list = [2**5, 2**6, 2**7]
# num_hidden_units_list = [2**9, 2**4, 2**5, 2**6]


# I am getting errors with 3 and 4 
# 1 is already done
users_list = [2, 5, 6, 7, 10]

# users_list = [1]



# dataprocessed = [True, False]


results_df_intra = pd.DataFrame(columns = ['model_name',
              'time_offset',
              'dim', 
              'batch_size', 
              'user', 
              "pca_knn_accuracy", 
              "cebra_knn_accuracy",
              "% _ difference",
              "cebra_outperform"])




def classificationGridSearch(user: int, 
                             results_df,
                             model_name:str, 
                             batch_size: int, 
                             iterations: int,
                             time_offset: int, 
                             dim: int):


    dataprocessed  = True

    if dataprocessed:
        emg_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'emg')
        emg_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'emg')
        emg_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'emg')

        glove_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'glove')
        glove_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'glove')
        glove_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'glove')

        restim_tr1 = auxf.getProcessedData(user = user, dataset = 1, type_data = 'training', mode = 'restimulus').astype(int)
        restim_tr2 = auxf.getProcessedData(user = user, dataset = 2, type_data = 'validation', mode = 'restimulus').astype(int)
        restim_test = auxf.getProcessedData(user = user, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)

    ## ------ GET DATA -------- ####


    ## ------ DEFINE MODEL -------- ####
        
    print('batch size in func', batch_size)

    cebra_model_def = CEBRA(
        model_architecture = model_name,
        batch_size= batch_size,
        temperature_mode='auto',
        #min_temperature=1,
        learning_rate = 0.0001,
        max_iterations = iterations,
        time_offsets = time_offset,
        output_dimension = dim, 
        device = "cuda_if_available",
        verbose = True,
        conditional='time_delta',
        distance = 'cosine',
        #num_hidden_units= num_hidden_unit,
    )


    ## ------ DEFINE MODEL -------- ####

    emg_tr_concat = np.concatenate([emg_tr1, emg_tr2])

   
   ### ------ CEBRA --------

    cebra_model = deepcopy(cebra_model_def)


    cebra_model.partial_fit(emg_tr1, glove_tr1, restim_tr1)
    cebra_model.partial_fit(emg_tr2, glove_tr2, restim_tr2)

    # cebra_model.save(f"./classification_gridsearch/models/{user}_{model_name}_{batch_size}_{time_offset}_{dim}_{num_hidden_unit}_{iterations}.pt")

    cebra_model.save(f"./classification_gridsearch/models/{user}_{model_name}_{batch_size}_{time_offset}_{dim}_{iterations}.pt")

    # emg_cebra_tr_concat = cebra_model.transform(emg_tr_concat)

    # emg_cebra_test = cebra_model.transform(emg_test)

    # restim_concat = np.concatenate([restim_tr1, restim_tr2])

    # knn_cebra = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')

    # knn_cebra.fit(emg_cebra_tr_concat, restim_concat)

    # knn_cebra_pred = knn_cebra.predict(emg_cebra_test)

    # cebra_knn_accuracy = accuracy_score(restim_test, knn_cebra_pred)


    # ### ------ PCA ---------- ###

    # pca = PCA(n_components= dim)

    # emg_pca_concat = pca.fit_transform(emg_tr_concat)

    # emg_pca_test = pca.transform(emg_test)

    # knn_pca = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')
    # knn_pca.fit(emg_pca_concat, restim_concat)

    # knn_pca_pred = knn_pca.predict(emg_pca_test)

    # pca_knn_accuracy = accuracy_score(restim_test, knn_pca_pred)

    # print("PCA ACC: ", pca_knn_accuracy)
    # print("CEBRA ACC", cebra_knn_accuracy)

    # difference = cebra_knn_accuracy - pca_knn_accuracy

    # if difference > 0:
    #     cebra_outpeform = True
    
    # else:
    #     cebra_outpeform = False


    # # --- APPEND TO DF ----- # 
        
    # print("batch size going into df", batch_size)


    # # df_row = {'model_name' : model_name,
    # #           'time_offset' : time_offset,
    # #           'dim' : dim, 
    # #           'batch_size' : batch_size, 
    # #           'num_hidden_units' : num_hidden_unit,
    # #           'user' : user, 
    # #           "pca_knn_accuracy" : pca_knn_accuracy, 
    # #           "cebra_knn_accuracy" : cebra_knn_accuracy,
    # #           "% _ difference" : difference*100,
    # #           "cebra_outperform" : cebra_outpeform}
    
    # df_row = {'model_name' : model_name,
    #         'time_offset' : time_offset,
    #         'dim' : dim, 
    #         'batch_size' : batch_size, 
    #         'user' : user, 
    #         "pca_knn_accuracy" : pca_knn_accuracy, 
    #         "cebra_knn_accuracy" : cebra_knn_accuracy,
    #         "% _ difference" : difference*100,
    #         "cebra_outperform" : cebra_outpeform}
    
    # df_row = pd.DataFrame(data=[df_row], index=[0])  # or any other index you prefer


    # results_df = pd.concat([results_df_intra, df_row])
    # # results_df.to_csv("./classification_gridsearch/results/results_df.csv")
    # results_df.to_csv("./todelete/test.csv")


    # return results_df

    return results_df

    
    # results_df = results_df.append(df_row, ignore_index = True




# total_len = len(dm_list) * len(users_list) * len(models_list) * len(batch_size_list) * len(time_offsets_list) * len(num_hidden_units_list) 

total_len = len(dm_list) * len(users_list) * len(models_list) * len(batch_size_list) * len(time_offsets_list)

iterations = 1
k = 201
progress = 0


for model_name in models_list:
    for user in users_list: 
        for batch_size in batch_size_list:
            for time_offset in time_offsets_list:
                #for num_hidden_unit in num_hidden_units_list:
                for dim in dm_list:

                    progress += 1

                    print(batch_size_list)

                    progress_percent = (progress / total_len) * 100

                    print(f"PROGRESS %: {progress_percent}")

                    print("batch size into func", batch_size)


                    results_df_intra = classificationGridSearch(user = user, 
                                                results_df=results_df_intra, 
                                                model_name=model_name, 
                                                batch_size=batch_size, 
                                                iterations = iterations, 
                                                time_offset=time_offset, 
                                                dim = dim)                    









directory_path = './classification_gridsearch/models'

model_paths = []

for filename in os.listdir(directory_path):
    if filename.endswith('.pt'):  # Check for PyTorch model files (or adjust the extension as needed)
        # Construct the full path and add it to the list
        full_path = os.path.join(directory_path, filename)
        model_paths.append(full_path)



# this script will test the capacity of the cebra models to test across different users (for now - user 1 against all other users)

k = 201

results_df_inter = pd.DataFrame(columns = ['model_name',
              'time_offset',
              'dim', 
              'batch_size', 
              'user_train',
              'user_test', 
              'iterations',
              "pca_knn_accuracy", 
              "cebra_knn_accuracy",
              "% _ difference",
              "cebra_outperform"])

user_test_list = [10, 9, 8, 7, 6]
user_train_list = [1, 2, 3, 4, 5, 6, 7, 10]



def runGridsearchUsers(results_df):


    for user_tr in user_train_list:
        for user_test in user_test_list:

            print(user_test)
            print(user_test_list)

            for model_path in model_paths:



                filename = model_path.split('/')[-1]
                
                # Remove the file extension
                filename_no_ext = filename.split('.')[0]

                # Split the filename into parts
                parts = filename_no_ext.split('_')

                user = parts[0]
                model_name = parts[1]
                batch_size = parts[2]
                time_offset = parts[3]
                dim = parts[4]
                iterations = parts[5]

                print("batch size", batch_size)
                print("time_offset" ,time_offset)
                print("dim", dim)

                emg_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'emg')
                emg_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'emg')

                glove_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'glove')
                glove_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'glove')

                restim_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'restimulus').astype(int)
                restim_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'restimulus').astype(int)


                emg_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'emg')
                glove_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'glove')
                restim_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)


                emg_tr_concat = np.concatenate([emg_tr1, emg_tr2])
                restim_tr_concat = np.concatenate([restim_tr1, restim_tr2])


                model = cebra.CEBRA.load(model_path)

                emg_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'emg')
                emg_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'emg')

                glove_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'glove')
                glove_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'glove')

                restim_tr1 = auxf.getProcessedData(user = user_tr, dataset = 1, type_data = 'training', mode = 'restimulus').astype(int)
                restim_tr2 = auxf.getProcessedData(user = user_tr, dataset = 2, type_data = 'validation', mode = 'restimulus').astype(int)


                

                emg_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'emg')
                glove_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'glove')
                restim_test = auxf.getProcessedData(user = user_test, dataset = 3, type_data = 'test', mode = 'restimulus').astype(int)


                emg_tr_concat = np.concatenate([emg_tr1, emg_tr2])
                restim_tr_concat = np.concatenate([restim_tr1, restim_tr2])

                print("dim", dim)
                dim = int(dim)
                
                model = cebra.CEBRA.load(model_path)

                training_embedding = model.transform(emg_tr_concat)
                test_embedding = model.transform(emg_test)

                knn_cebra = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')

                knn_cebra.fit(training_embedding, restim_tr_concat)

                knn_cebra_pred = knn_cebra.predict(test_embedding)

                cebra_knn_accuracy = accuracy_score(restim_test, knn_cebra_pred)

                pca = PCA(n_components= dim)

                emg_pca_train = pca.fit_transform(emg_tr_concat)

                emg_pca_test = pca.transform(emg_test)

                knn_pca = cebra.KNNDecoder(n_neighbors=k, metric = 'cosine')

                knn_pca.fit(emg_pca_train, restim_tr_concat)

                knn_pca_pred = knn_pca.predict(emg_pca_test)

                pca_knn_accuracy = accuracy_score(restim_test, knn_pca_pred)

                difference = cebra_knn_accuracy - pca_knn_accuracy

                if difference > 0:
                    cebra_outpeform = True
                
                else:
                    cebra_outpeform = False


                df_row = {'model_name' : model_name,
                'time_offset' : time_offset,
                'dim' : dim, 
                'batch_size' : batch_size, 
                'user_train' : 1,
                'user_test' : user_test, 
                'iterations' : iterations,
                "pca_knn_accuracy" : pca_knn_accuracy, 
                "cebra_knn_accuracy" : cebra_knn_accuracy,
                "% _ difference" : difference*100,
                "cebra_outperform" : cebra_outpeform}

                    
                df_row = pd.DataFrame(data=[df_row], index=[0])  # or any other index you prefer


                results_df = pd.concat([results_df, df_row])
                results_df.to_csv("./classification_gridsearch/across_users/results_df.csv")


                print("PCA ACC: ", pca_knn_accuracy)
                print("CEBRA ACC", cebra_knn_accuracy)

    return results_df



runGridsearchUsers(results_df_inter)





































































