import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
from math import exp, expm1,log
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import interp
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
#from sklearn.cross_validation import train_test_split
from pycm import *

def valuation(class_dict):    
    roc=[]
    for j in range(5):
        temp=[]
        for i in range(5):
            temp.append(class_dict[i][j])
        roc.append(temp)
    mean_class=[]
    std_class=[]
    
    for k in range(len(roc)):
        mean_class.append(np.mean(roc[k]))
        std_class.append(np.std(roc[k]))
    return np.around(mean_class,decimals=4),np.around(std_class,decimals=4)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('D:/OneDrive/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/FS/Smoking_DataSheet_5s_FS.csv')
df.shape
df.isnull().any()

df=df.dropna()

clf1 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', probability = True)
clf4 = RandomForestClassifier(max_depth=40, random_state=0)
clf5 = LogisticRegression(random_state=0)
clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
sm = SMOTE(random_state=42)

kf = StratifiedKFold(n_splits=5, shuffle = True, random_state=45) 
arr = df.values
X_All = arr[0:,0:len(df.columns)-1]
y = arr[:,len(df.columns)-1]
sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)

it=[clf1,clf2,clf3,clf4,clf5,clf6]
k=1
for m in it:
    auc=[]
    f1=[]
    accuracy=[]
    i=0
    for train_index,test_index in kf.split(X,y):    
        i=i+1
        X_train1, X_test = X[train_index],X[test_index]
        y_train1, y_test = y[train_index],y[test_index]
        X_train, y_train = sm.fit_sample(X_train1, y_train1)          
        m.fit(X_train1, y_train1)
        ypred = m.predict(X_test)
        cm = ConfusionMatrix(actual_vector=y_test, predict_vector=ypred)
#        with open('KNNreport_'+str(i)+'.txt', 'w') as f:
#            print(cm, file=f)
#            f.close()
        
        auc.append(cm.AUC)
        accuracy.append(cm.ACC)
        f1.append(cm.F1)
    df_accu=pd.DataFrame(valuation(accuracy))
    df_auc=pd.DataFrame(valuation(auc))
    df_f1=pd.DataFrame(valuation(f1))
    final=pd.concat([df_accu, df_auc,df_f1], axis=1)
    final.to_csv('D:/OneDrive/Quit Smoking Work/Final Experiment-Shared_Folder/Results/FS Results/s5/'+str(k)+'model.csv')
    k=k+1



