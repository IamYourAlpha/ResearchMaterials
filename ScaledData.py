import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv('CompleteSet.csv')
feature = ['solvedUser', 'AC', 'TLE', 'MLE', 'WA', 'RTE']
data = data[feature]
print data.head()
x =  np.array(data)

pca = decomposition.PCA()
pca.fit(x)

print pca
print pca.explained_variance_
pca.n_components = 2
x_reduced = pca.fit_transform(x)

kmeans = KMeans(n_clusters=3, max_iter=500, n_init=10, random_state=None, init='k-means++')
y_pred = kmeans.fit_predict(x_reduced)
print y_pred
col = np.unique(y_pred)
plt.scatter(x_reduced[:,0], x_reduced[:,1], marker='o', c = y_pred)
data = pd.read_csv('CompleteSet.csv')
data['pred'] = y_pred
print data[ ['AC', 'pred']].groupby('pred').mean()
print data[ ['WA', 'pred']].groupby('pred').mean()
print data['cat'].value_counts()
#print data['cat'].hist()
#data.to_csv('with_cluster_group.csv',  encoding='utf-8')
plt.show()
