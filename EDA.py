import pandas as pd
import numpy as np


data = pd.read_csv('D:/OneDrive/PHD Work/Quit Smoking Work/Final Experiment-Shared_Folder/Models - Data Sheets/GM/Smoking_DataSheet_5s_GM.csv')

look = data.head(20) ## Peek into the Top 20 rows of the Data to get an idea of data.
print(look)

data.shape          # To know the number of Rows and Columns of the Data
data.columns        # To List the Individual Columns

data_type = data.dtypes # To checck the Data Type of the variables
print(data_type)

##############################################################################################################
# Below Code snippet will walk us through the Descriptive Statistical Summary of the Raw Data 

description = data.describe()
print(description)
description.to_csv("D:\OneDrive\PHD Work\Quit Smoking Work\Final Experiment-Shared_Folder\Results\EDA Analysis\EDA_GM_5s.csv", index=False) # The results are transported to a csv file for better view

data.isnull().any()