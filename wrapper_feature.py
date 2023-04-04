# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:16:47 2019

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
 

data = pd.read_csv('D:/OneDrive/Quit Smoking Work/Final Experiment-Shared_Folder/fs/s1.csv', header=0) 
data=data.dropna()

arr = data.values
X = arr[0:,0:len(data.columns)-2]
y = arr[:,len(data.columns)-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs = -1),
#         k_features = 40,
#          forward= True,
#          floating = False,
#          verbose= 2,
#          scoring= 'accuracy',
#          cv = 4,
#          n_jobs= -1
#         ).fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)

sfs1 = SFS(estimator=knn, 
           k_features=(1, 86),
           forward=False, 
           floating=False, 
           scoring='accuracy',
           cv=5)

pipe = make_pipeline(StandardScaler(), sfs1)
pipe.fit(X_train, y_train)
print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)

plot = plot_sfs(sfs1.get_metric_dict(), kind='std_err',figsize=(16,9));
plt.title('Sequential Backward Selection')
plt.grid()
plt.set_xticklabels(['0',	'5',	'10',	'15',	'20',	'25',	'30',	'35',	'40',	'45',	'50',	'55',	'60',	'65',	'70',	'75',	'80',	'85',	'90'])
plt.savefig('D:/OneDrive/Quit Smoking Work/Final Experiment-Shared_Folder/fs/Fig_s1.png', dpi=300)

