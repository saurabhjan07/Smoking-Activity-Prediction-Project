# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:28:23 2020

@author: Pradeep Poddar
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime 
from sklearn.preprocessing import StandardScaler 
from pycm import *

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
                 print_cm=True, cm_cmap=plt.cm.Greens):
    
    # to store results at various phases
    results = dict()
    # time at which model starts training 
    print('training the model..')
    train_start_time = datetime.now()
    model.fit(X_train, y_train)
    train_end_time = datetime.now()
    print('Done \n \n')
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    
    # predict test data - Out of Sample Testing
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred_ost = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted_ost'] = y_pred_ost
   

    # predict test data - In Sample Testing
    print('Predicting train data')
    test_start_time = datetime.now()
    y_pred_ist = model.predict(X_train)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted_ist'] = y_pred_ist
   
    # calculate overall accuracty of the model
    #accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    #results['accuracy'] = accuracy
    #print('---------------------')
    #print('|      Accuracy      |')
    #print('---------------------')
    #print('\n    {}\n\n'.format(accuracy))
    
#................................................................................................................................
    
### Out of Sample Testing
    
    # confusion matrix_ost
    cm_ost = metrics.confusion_matrix(y_test, y_pred_ost)
    results['confusion_matrix_ost'] = cm_ost
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm_ost))
    
    
    # plot confusin matrix
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm_ost, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report Out of Sample Testing |')
    print('-------------------------')
    classification_report_ost = metrics.classification_report(y_test, y_pred_ost)
    # store report in results
    results['classification_report_ost'] = classification_report_ost
    print(classification_report_ost)
    
    # add the trained  model to the results
    results['model'] = model
    
#...................................................................................................................................
    
#### In Sample tetsing
    
    # confusion matrix_ist
    cm_ist = metrics.confusion_matrix(y_train, y_pred_ist)
    results['confusion_matrix'] = cm_ist
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm_ist))
    
    
    # plot confusin matrix
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm_ist, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report In Sample tetsing|')
    print('-------------------------')
    classification_report_ist = metrics.classification_report(y_train, y_pred_ist)
    # store report in results
    results['classification_report_ist'] = classification_report_ist
    print(classification_report_ist)
    
    # add the trained  model to the results
    #results['model'] = model
    
    
    return results
   
def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('--------------------------')
    print('|     Best parameters     |')
    print('--------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


    #  number of cross validation splits
    print('---------------------------------')
    print('|   No of CrossValidation sets   |')
    print('--------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))
    
    
    
    
from sklearn import linear_model
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def logistic():
    from sklearn.linear_model import LogisticRegression
    parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
    log_reg = LogisticRegression()
    log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
    log_reg_grid_results =  perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(log_reg_grid_results['model'])

def svm():
    #from sklearn.svm import SVC
    from sklearn import svm 
    parameters = {'C':[0.001, 0.01, 0.1, 1, 10], 'gamma': [ 0.001, 0.01, 0.1, 1], 'kernel':('linear', 'poly', 'rbf', 'sigmoid')} # 'C':[2,8,16], 'gamma': [ 0.0078125, 0.125, 2]
    rbf_svm =  svm.SVC() # SVC(kernel='rbf') #
    rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, cv=5, n_jobs=-1)
    rbf_svm_grid_results = perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(rbf_svm_grid_results['model'])

def rf():
    from sklearn.ensemble import RandomForestClassifier
    params = {'bootstrap': [True, False], 'max_depth': [10, 20, 30, 40, 50],'n_estimators': [50, 100, 500, 1000]}
#              'max_features': ['auto', 'sqrt'],\
#              'min_samples_leaf': [1, 2, 4],\
#              'min_samples_split': [2, 5, 10],\
              
    #{'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)} 
    rfc = RandomForestClassifier()
    rfc_grid = GridSearchCV(rfc, param_grid=params, cv=5, n_jobs=-1)
    rfc_grid_results = perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(rfc_grid_results['model'])

def dt():
    from sklearn.tree import DecisionTreeClassifier
    parameters = {'criterion':['gini','entropy'],'max_depth': np.arange(3, 40)} 
    dt = DecisionTreeClassifier()
    dt_grid = GridSearchCV(dt,param_grid=parameters, cv=5, n_jobs=-1)
    dt_grid_results = perform_model(dt_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(dt_grid_results['model'])

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], max_depth = [4,6,8,12], criterion = ['gini', 'entropy']}

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    parameters = {'n_neighbors': np.arange(1,15), 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn,param_grid=parameters, cv=5, n_jobs=-1)
    knn_grid_results = perform_model(knn_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(knn_grid_results['model'])

def adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    parameters = {'n_estimators':[50, 100, 500, 1000, 2000], 'learning_rate':[.001,0.01,.1,1]}
    adaboost = AdaBoostClassifier()
    adaboost_grid = GridSearchCV(adaboost,param_grid=parameters, cv=5, n_jobs=-1)
    adaboost_grid_results = perform_model(adaboost_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    print_grid_search_attributes(adaboost_grid_results['model'])

#AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
    
    
#KNeighborsClassifier(n_neighbors=3)

### .............................................. ### ............................................... ### ........................................

labels=["Downstairs", "Upstairs","Walking","Running","Smoking"]

df_GM_1s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_1s_GM.csv')
df_GM_3s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_3s_GM.csv')
df_GM_5s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_5s_GM.csv')

df_FS_1s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_1s_FS.csv')
df_FS_3s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_3s_FS.csv')
df_FS_5s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_5s_FS.csv')

df_it = [df_GM_1s, df_GM_3s, df_GM_5s, df_FS_1s, df_FS_3s, df_FS_5s]

for df in df_it:
    print(df.shape)
    df.isnull().any()
    df=df.dropna()
    
    arr = df.values
    X = arr[0:,0:len(df.columns)-1]
    y = arr[:,len(df.columns)-1]
    
    sc_x=StandardScaler()
    X = sc_x.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)   
    
    print("Main Dataframe Shape:",df.shape)
    print("Train Dataframe Shape:",X_train.shape)
    print("Test Dataframe Shape:",X_test.shape)
    print("Train Labels Shape:",y_train.shape)
    print("Test Labels Shape:",y_test.shape)

    logistic()
    svm()
    rf()
    dt()
    knn()
    adaboost()
    knn

#xgboost
with open('KNNreport_'+str(i)+'.txt', 'w') as f:
#            print(cm, file=f)
#            f.close()


#### ..................................... ROUGH WORK .................................................................

df = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_5s_GM.csv')

cols = ['std0',	'ptp0',	'mad0',	'root mean square0',	'kurtosis0',	'quartile0',	'median0',	'amp0',	'energy0',	'frequency entropy0',	'std1',	'ptp1',	'root mean square1',	'skew1',	'quartile1',	'median1',	'amp1',	'energy1',	'frequency entropy1',	'mean2',	'std2',	'ptp2',	'mad2',	'root mean square2',	'skew2',	'median2',	'amp2',	'energy2',	'std3',	'ptp3',	'root mean square3',	'energy3',	'root mean square4',	'skew4',	'amp4',	'frequency entropy4',	'skew5',	'frequency entropy5',	'corr01',	'corr02',	'corr04',	'corr05',	'corr14',	'corr15', 'Class']

print(len(cols))
print(df_GM_1s.shape)
df = df_GM_1s.drop(columns=[col for col in df_GM_1s if col not in cols])
print(df.shape)
df.to_csv("D:/OneDrive/Smoking_DataSheet_1s_FS.csv", index=False)