import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Problems_csv.csv')
#data = data[ data.dtypes[ data.dtypes == "float64" | data.dtypes=="int64" ].index.values]
dataNum = data[ data.dtypes[ data.dtypes == "int64" ].index.values]
minmax = MinMaxScaler()
kmeans = cluster.KMeans(n_clusters=6, max_iter=500, random_state=None, init='k-means++')
feature = ['solvedUser', 'submissions']
scaled_data = minmax.fit_transform(dataNum[feature])
y_pred =  kmeans.fit_predict(scaled_data)
dataNum['pred'] = y_pred
cluster_1_data = dataNum.loc[ dataNum['pred'] == 0]
print cluster_1_data
#print kmeans.cluster_centers_
print np.unique(y_pred)
#plt.scatter(dataNum['solvedUser'], dataNum['submissions'], c=y_pred)
plt.xlabel("solvedUser")
plt.ylabel("submissions")
center = kmeans.cluster_centers_
label = kmeans.labels_
print label
color = []
import random
color = ['r','g','b']
#plt.scatter(center[:,0], center[:,1],marker="x", s=150, linewidths=5,c=color)
dataNum['pred'].hist(bins=50)
plt.show()
