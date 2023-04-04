'''
@Saurabh 13 July 2020

This model is developed while submitting in SI Journal of Biomedical informatics

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from datetime import datetime 
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
from math import exp, expm1,log
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import interp
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split
from pycm import *
from sklearn.model_selection import train_test_split

df_GM_1s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_1s_GM.csv')
df_GM_3s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_3s_GM.csv')
df_GM_5s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_5s_GM.csv')

df_FS_1s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_1s_FS.csv')
df_FS_3s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_3s_FS.csv')
df_FS_5s = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_5s_FS.csv')

df_it = [df_GM_1s, df_GM_3s, df_GM_5s, df_FS_1s, df_FS_3s, df_FS_5s]

# df = df_FS_5s # clf = clf_dt # df.shape #len(X) len(y) len(y_test)

clf_adaboost = AdaBoostClassifier(n_estimators=1000, learning_rate=1, random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=5, metric = 'manhattan')
clf_svm = svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.01)
clf_rf = RandomForestClassifier(max_depth=30, n_estimators = 1000, random_state=0)
clf_lr = LogisticRegression(C=20, penalty = 'l2')
clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth = 21)

clf_it = [clf_adaboost, clf_knn, clf_svm, clf_rf, clf_lr, clf_dt]


df_label = ['GM_1s','GM_3s','GM_5s','FS_1s','FS_3s','FS_5s']
clf_label = ['adaboost', 'knn', 'svm', 'rf', 'lr', 'dt']

i=0
for df in df_it:
    print(df.shape)
    df.isnull().any()
    df=df.dropna()
    
    arr = df.values
    X = arr[0:,0:len(df.columns)-1]
    y = arr[:,len(df.columns)-1]
    
    sc_x=StandardScaler()
    X = sc_x.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)   
    
    print(df_label[i])
    
#    print("Main Dataframe Shape:",df.shape)
#    print("Train Dataframe Shape:",X_train.shape)
#    print("Test Dataframe Shape:",X_test.shape)
#    print("Train Labels Shape:",y_train.shape)
#    print("Test Labels Shape:",y_test.shape)
    
    j = 0
    for clf in clf_it:
        
        print(clf_label[j])
        
        t1 = datetime.now()
        clf.fit(X_train, y_train)
        t2 = datetime.now()
        T =  t2 - t1
        print('Training_Time:',T.seconds)
        
        # In Sample Testing
        
        t1 = datetime.now()
        ypred_ist = clf.predict(X_train)
        t2 = datetime.now()
        T =  t2 - t1
        print('Testing_Time_InSample:',T.seconds)
        
        
#        cm_ist = ConfusionMatrix(actual_vector=y_train, predict_vector=ypred_ist)
#        with open(df_label[i]+'_'+clf_label[j]+'_IST_CM.txt', 'w') as f:
#            print(cm_ist, file=f)
#            f.close()
#            
#        clf_report_ist = metrics.classification_report(y_train, ypred_ist)
#        with open(df_label[i]+'_'+clf_label[j]+'_IST_Report.txt', 'w') as f:
#            print(clf_report_ist, file=f)
#            f.close()

        # Out of Sample Testing
        
        t1 = datetime.now()
        ypred_ost = clf.predict(X_test)
        t2 = datetime.now()
        T =  t2 - t1
        print('Testing_Time_OutofSample:',T.seconds)
        
#        cm_ost = ConfusionMatrix(actual_vector=y_test, predict_vector=ypred_ost)
#        with open(df_label[i]+'_'+clf_label[j]+'_OST_CM.txt', 'w') as f:
#            print(cm_ost, file=f)
#            f.close()
#            
#        clf_report_ost = metrics.classification_report(y_train, ypred_ist)
#        with open(df_label[i]+'_'+clf_label[j]+'_OST_Report.txt', 'w') as f:
#            print(clf_report_ost, file=f)
#            f.close()  
        
        j = j+1
    
    i = i+1
    print(" ")
#################################################################################################################################################