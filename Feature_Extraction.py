# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:07:17 2019

@author: RMSOEE
"""

import os
import pandas as pd
import numpy as np
from window_slider import Slider
import itertools
from scipy.stats import skew
from scipy.stats import pearsonr
from scipy.fftpack import fft
from scipy import signal
from statsmodels import robust
from scipy.stats import kurtosis as krt
from scipy.stats import iqr
from pywt import wavedec
 
rootdir = 'D:\OneDrive\Quit Smoking Work\Final Experiment-Shared_Folder\Full Data'
final_df = pd.DataFrame()
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        df = pd.read_csv(os.path.join(subdir, file), sep=",", header=None)
        splitResult = file.split( "_" )
        Activity = splitResult[0]
        if (df.shape[1] == 8):
            df = df.drop(df.columns[[0, 1]], axis=1)
        else:
            df = df.drop(df.columns[[0]], axis=1)                
        col = df.columns.values
        new_df = pd.DataFrame()
        
        for i in range(len(col)):
            
            mean=[]
            std_dev=[]
            ptp=[]
            root_mean_sqr=[]
            mad=[]
            skewness=[]
            kurtosis=[]
            quartile=[]
            media=[]
            set1=[]
            amp=[]
            energy=[]
            frequency_entropy=[]
            val=np.asarray(df[col[i]].tolist()).astype(np.float) 
            bucket_size = 70
            overlap_count =35
            slider = Slider(bucket_size,overlap_count)
            slider.fit(val) 
            
            while True:
                window_data = slider.slide()
                mean.append(np.mean(window_data))
                std_dev.append(np.std(window_data))
                ptp.append(np.ptp(window_data))
                mad.append(robust.mad(window_data))
                root_mean_sqr.append(np.sqrt((np.sum(np.power(window_data,2)))/bucket_size))
                skewness.append(skew(window_data))
                kurtosis.append(krt(window_data))
                quartile.append(iqr(window_data))
                media.append(np.median(window_data))
#                cof1=wavedec(window_data, 'db3', level=1)
#                set1.append(np.sum(np.square(cof1[0])))
                amp.append(abs(fft(window_data))[0])
                energy.append(np.sum(np.square(abs(fft(window_data)))))
                pi=(signal.welch(window_data)[1])/np.sum(signal.welch(window_data)[1])
                entropy=-np.sum(pi*(np.log(pi)))
                frequency_entropy.append(entropy)
                if slider.reached_end_of_list(): break
            
            new_df['mean'+str(i)+'']=mean
            new_df['std'+str(i)+'']=std_dev
            new_df['ptp'+str(i)+'']=ptp
            new_df['mad'+str(i)+'']=mad
            new_df['root mean square'+str(i)+'']=root_mean_sqr
            new_df['skew'+str(i)+'']=skewness
            new_df['kurtosis'+str(i)+'']=kurtosis
            new_df['quartile'+str(i)+'']=quartile
            new_df['median'+str(i)+'']=media
            new_df['amp'+str(i)+'']=amp
            new_df['energy'+str(i)+'']=energy
            new_df['frequency entropy'+str(i)+'']=frequency_entropy
        
        a = list(itertools.combinations(range(len(col)),2))
        for j in range(len(a)):
            corr=np.transpose(df.iloc[0:,np.asarray(a[j])].values)
            slider = Slider(bucket_size,overlap_count)
            slider.fit(corr)
            pearson=[]       
            while True:
                window_data = slider.slide()
                pearson.append(pearsonr(window_data[0],window_data[1])[0])
                if slider.reached_end_of_list(): break
            new_df['corr'+str(a[j][0])+''+str(a[j][1])+'']=pearson
        
        if (Activity == "Downstairs"):
            new_df['Class'] = 0
        if (Activity == "Upstairs"):
            new_df['Class'] = 1
        if (Activity == "Walking"):
            new_df['Class'] = 2
        if (Activity == "Running"):
            new_df['Class'] = 3
        if (Activity == "Smoking"):
            new_df['Class'] = 4
            
        final_df = final_df.append(new_df, ignore_index = True)
                 
        #except TypeError:
            #check=None
            
/       
