import pandas as pd
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

data = pd.read_csv('dumy.csv')
feature = ['solvedUser', 'AC', 'WA']
print data[feature].head()
fignum = 1
data = data[feature]
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=500, random_state=None)
fig = plt.figure(fignum, figsize = (4,3))
ax = Axes3D(fig, rect = [0,0,0.95, 1], elev=48, azim=134)
kmeans.fit(data)
labels = kmeans.labels_
ax.scatter(data['WA'], data['AC'], data['solvedUser'], c = labels.astype(np.float), edgecolor='k')
plt.show()
