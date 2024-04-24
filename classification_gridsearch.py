import sys
sys.path.append("./BEng_dissertation")


import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import aux_functions as auxf
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch import nn
import cebra.models
import cebra.data
import os
import pandas as pd
from torch import nn
from copy import deepcopy
import seaborn as sns
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump
from joblib import load 




params = {
    "font.family": "serif",
}


import matplotlib as mpl
mpl.rcParams.update(params)

iterations = 20000

user_list = np.arange(1, 13)
gesture_list = np.arange(1, 10)

best_model_list = []


user_results = []

def runGridSearch():

    for user in user_list: 

        user_row = []

        emg1 = auxf.getProcessedEMG(user = user, dataset= 1, type_data = 'all')
        emg2 = auxf.getProcessedEMG(user = user, dataset= 2, type_data = 'all')




        stim1 = auxf.getProcessedData(user = user, dataset=1, mode = 'restimulus', rawBool=False)
        stim2 = auxf.getProcessedData(user = user, dataset=2, mode = 'restimulus', rawBool=False)



        glove1 = (auxf.getMappedGlove(user = user, dataset=1)).T
        glove2 = (auxf.getMappedGlove(user = user, dataset=2)).T




        # create trainval
        emg_trainval = np.concatenate((emg1, emg2))
        stim_trainval = np.concatenate((stim1, stim2))
        glove_trainval = np.concatenate((glove1, glove2))




        pipeline_cebra = Pipeline([
            ('cebra', CEBRA(
                    # model_architecture = 'offset10-model',
                    batch_size= 256,
                    temperature_mode='auto',
                    learning_rate = 0.0001,
                    max_iterations = iterations,
                    time_offsets = 25,
                    output_dimension = 3, 
                    device = "cuda_if_available",
                    min_temperature = 0.3,
                    verbose = True,
                    conditional='time_delta', 
                    distance = 'cosine' 
                )   ), 
            ('knn', KNN())
        ])


        pipeline_lda = Pipeline([
            ('lda', LDA(n_components=3)), 
            ('knn', KNN())
        ])


        param_grid_cebra = {

            'knn__n_neighbors': [200, 350, 500],
            'cebra__model_architecture' : ['offset10-model', 'offset36-model'],
        }

        param_grid_lda = {

            'knn__n_neighbors': [200, 350, 500]

        }



        emg_test = auxf.getProcessedEMG(user = user, dataset= 3, type_data = 'all')
        stim_test = auxf.getProcessedData(user = user, dataset=3, mode = 'restimulus', rawBool=False)
        glove_test = (auxf.getMappedGlove(user = user, dataset=3)).T


        X_train = emg_trainval
        y_train = stim_trainval


        grid_search_cebra = GridSearchCV(pipeline_cebra, param_grid_cebra, cv=3, scoring='accuracy')
        grid_search_cebra.fit(X_train, y_train)

        grid_search_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=3, scoring='accuracy')
        grid_search_lda.fit(X_train, y_train)


        best_params_cebra = grid_search_cebra.best_params_
        best_score_cebra = grid_search_cebra.best_score_

        best_params_lda = grid_search_lda.best_params_
        best_score_lda = grid_search_lda.best_score_

        best_model_cebra = grid_search_cebra.best_estimator_
        best_model_lda = grid_search_lda.best_estimator_

        # Evaluate on test set
        test_score_cebra = accuracy_score(stim_test, best_model_cebra.predict(emg_test))
        test_score_lda = accuracy_score(stim_test, best_model_lda.predict(emg_test))
        
        # Gather all results in a list of dictionaries for easy DataFrame construction
        user_row = {
            'user': user,
            'CEBRA_acc': test_score_cebra,
            'LDA_acc': test_score_lda,
            'best_params_cebra': grid_search_cebra.best_params_,
            'best_params_lda': grid_search_lda.best_params_,
        }
        user_results.append(user_row)
        
        # Store the best models
        best_model_list.append((best_model_cebra, best_model_lda))


    classif_results_df = pd.DataFrame(user_results, columns = ['user', 'CEBRA_acc', 'LDA_acc', 'best_params_cebra', 'best_params_lda'])

    classif_results_path = f"./classification/results"
    auxf.ensure_directory_exists(classif_results_path)

    classif_results_df.to_csv(f"{classif_results_path}/classif_results.csv")



# --- run classification ---- -#

runGridSearch()


# ----- process the classification results ------ # 

classif_results_path = f"./classification/results"
   
results = pd.read_csv(f"{classif_results_path}/classif_results.csv")


# ----- bar plot --------- #

fig, ax = plt.subplots(figsize=(18, 12))


sns.barplot(x=results['user'], y=results['CEBRA_acc'], color='orangered', label='CEBRA', alpha=0.8, ax=ax)
sns.barplot(x=results['user'], y=results['LDA_acc'], color='royalblue', label='LDA', alpha=0.5, ax=ax) 

ax.legend(fontsize = 17)

ax.set_xticklabels([f'User {i}' for i in range(1, 13)], fontsize = 17)
ax.set_title("Accuracy of KNN with CEBRA and LDA embeddings", fontsize = 20)
ax.set_ylabel("Accuracy", fontsize = 19)
ax.set_xlabel('')
ax.set_ylim((0, 1))
# plt.savefig(f"{classif_results_path}/classif_results_plot.png")


# ----- text file : stats ------- # 

with open(f"{classif_results_path}/classif_result_stats.txt", 'w') as file:
    file.write(f"Mean accuracy CEBRA embeddings: {results['CEBRA_acc'].mean()} \n")
    file.write(f"Std accuracy CEBRA embeddings: {results['CEBRA_acc'].std()} \n")
    file.write(f"Mean accuracy LDA embeddings: {results['LDA_acc'].mean()} \n")
    file.write(f"Std accuracy LDA embeddings: {results['LDA_acc'].std()} \n")

# ----- improved plot ------- #


melted_results = results.melt(id_vars=['user'], 
                              value_vars=['CEBRA_acc', 'LDA_acc'],
                              var_name='method', 
                              value_name='score')

melted_results['Dim. Reduction'] = melted_results['method'].replace({'CEBRA_acc': 'CEBRA', 'LDA_acc': 'LDA'})


g = sns.catplot(data=melted_results, kind="bar",
                x="user", y="score", hue="Dim. Reduction",
                palette="muted", alpha=.6, height = 15, aspect=20/15)

# Set spines to show all sides of the plot
for ax in g.axes.flat:
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    ax.tick_params(axis='x', labelsize=22)  # Adjust the value as needed


    # Increase font size for tick labels
    ax.tick_params(labelsize=22)

# Customizations
g.set_axis_labels("User", "Accuracy", fontsize=22)
g.set_titles("")

# Adjusting the legend
new_labels = ["CEBRA", "LDA"]
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)
    t.set_fontsize(22)  # Increase font size for legend labels

# Set the ylim
g.set(ylim=(0, 1))
g._legend.set_bbox_to_anchor((0.99, 0.5))  
plt.setp(g._legend.get_title(), fontsize=20)  # Increase font size for legend title

plt.savefig(f"{classif_results_path}/classif_results_plot.png")
