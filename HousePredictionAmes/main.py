import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda import memory_usage


#Here is where we load in the data as the training and testing .csv datasets
filePathTrain = r"\\wsl.localhost\Ubuntu\home\itsmejimmie\MachineLearningProjects\HousePredictionAmes\train.csv"
filePathTest = r"\\wsl.localhost\Ubuntu\home\itsmejimmie\MachineLearningProjects\HousePredictionAmes\test.csv"

#Then we have to read the .csv files to be able to call anything with them
dataTrain = pd.read_csv(filePathTrain)
dataTest = pd.read_csv(filePathTest)

#Then we're going to want to look at the information/statistics of the training set. (It's already assumed in the Kaggle
#dataset for testing that it's a good distribution meaning no missing items etc.)
dataTrain.info(memory_usage = "false")
print("\n")

#Then I want to print out the feature correlation information
corr_matrix = dataTrain.corr(numeric_only = True)
print(corr_matrix["SalePrice"].sort_values(ascending=False))






#numeric_cols = dataTrain.select_dtypes(include=[np.number])
#plt.figure(figsize=(10, 8))
#sns.heatmap(numeric_cols.corr(), cmap="RdBu_r")
#plt.title("Correlations Between Variables", size=15)
#plt.show()