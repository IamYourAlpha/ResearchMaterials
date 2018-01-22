###### Importing all neseccary librariese ########

import numpy as np
import pandas as pd
import json
import csv
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics 
from sklearn.decomposition import PCA
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skfuzzy as fuzz
#######################################################


data = pd.read_csv('with_cluster_group.csv')
related_features = ['AC', 'TLE', 'WA', 'solvedUser', 'submissions']
data_np  = np.array(data[related_features])

## Scale the value
data_np = preprocessing.StandardScaler().fit_transform(data_np)


########## Performing Princial Component analysis for dimension reduction
pca = PCA(n_components=2)
transformed_data = pca.fit(data_np).transform(data_np)


################################################################



############Begin Clustering ########################

kmeans = cluster.KMeans(n_clusters=4, max_iter=600, n_init=10, random_state=10, init='k-means++')
k_labels = kmeans.fit_predict(transformed_data)

db = cluster.DBSCAN(eps=0.3, min_samples=7)
d_labels = db.fit_predict(transformed_data)

Agglo = cluster.AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
aglo_labels = Agglo.fit_predict(transformed_data)


Agglo_comp = cluster.AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
aglocomp_labels = Agglo.fit_predict(transformed_data)

Birch = cluster.Birch(threshold=0.4, branching_factor=50, n_clusters=4, compute_labels=True)
birch_labels = Birch.fit_predict(transformed_data)


#fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(7,10))

print metrics.silhouette_score(transformed_data, k_labels)
print metrics.silhouette_score(transformed_data, birch_labels)
print metrics.silhouette_score(transformed_data, aglo_labels)
print metrics.silhouette_score(transformed_data, aglocomp_labels)
##
print metrics.calinski_harabaz_score(transformed_data, k_labels)
print metrics.calinski_harabaz_score(transformed_data, birch_labels)
print metrics.calinski_harabaz_score(transformed_data, aglo_labels)
print metrics.calinski_harabaz_score(transformed_data, aglocomp_labels)



plt.subplot(2,2,1)
plt.scatter(transformed_data[:,0], transformed_data[:,1], s=30, lw=0, alpha=0.7, c=k_labels, edgecolor='k')
plt.title('Kmeans++')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='o', c="white", alpha=1, s=40)
plt.subplot(2,2,2)
plt.title('Complete')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.scatter(transformed_data[:,0], transformed_data[:,1], s=30, lw=0, alpha=0.7, c=aglocomp_labels, edgecolor='k')
plt.subplots_adjust(top=.95, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.subplot(2,2,3)
plt.title('Ward')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.scatter(transformed_data[:,0], transformed_data[:,1], s=30, lw=0, alpha=0.7, c=aglo_labels, edgecolor='k')
plt.subplot(2,2,4)
plt.title('Birch')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.scatter(transformed_data[:,0], transformed_data[:,1], s=30, lw=0, alpha=0.7, c=birch_labels, edgecolor='k')
plt.subplots_adjust(top=.95, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

plt.show()

data['k_preds'] = k_labels
data['birch_labels'] = birch_labels
data['w_preds'] = aglo_labels
data['comp_preds'] = aglocomp_labels
print np.unique(kmeans.labels_)
'''for c in range(4):
    print 'Max and min AC',data[ data['k_preds']==c]['AC'].max(),data[ data['k_preds']==c]['AC'].min()
    print 'Max and min Wa',data[ data['k_preds']==c]['WA'].max(),data[ data['k_preds']==c]['WA'].min()
    print 'Max and min TLE',data[ data['k_preds']==c]['TLE'].max(),data[ data['k_preds']==c]['TLE'].min()
    print data[ data['k_preds']==c]['solvedUser'].mean()/data[ data['k_preds']==c]['submissions'].mean()
    print '*'*40
    '''
#fig, ax = plt.subplots(2,2)
#ax[0,0].hist(np.array(k_labels), bins=4)
#x[0,1].hist(np.array(birch_labels), bins=4)
#ax[1,0].hist(np.array(aglo_labels), bins=4)
#ax[1,1].hist(np.array(aglocomp_labels), bins=4)
#plt.show()




################### Fuzzy clustering begins ######################

fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((transformed_data[:,0],transformed_data[:,1]))
fpcs = []
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
x = transformed_data[:,0]
y = transformed_data[:,1]
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(x[cluster_membership == j],
                y[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')
    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()
plt.show()


fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
plt.show()


cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, 5, 2, error=0.005, maxiter=1000)
fig2, ax2 = plt.subplots()
ax2.set_title('Fuzzy clustering with 5 Clusters')
ax2.set_xlabel('Principal Component-1')
ax2.set_ylabel('Principal Component-2')

for j in range(5):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j ],
             alldata[1, u_orig.argmax(axis=0) == j ], 'o',
             label = 'Cluster ' + str(j))
for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')
ax2.legend()
plt.show()
fig, ax = plt.subplots()
data['FuzzPredict'] =  np.argmax(u_orig, axis=0)
N, bins, patches = ax.hist(np.array(data['FuzzPredict']), bins=5)
ax.set_label('Fuzzy')
plt.show()
for c in range(5):
    print 'Max and min AC',data[ data['FuzzPredict']==c]['AC'].max(),data[ data['FuzzPredict']==c]['AC'].min()
    print 'Max and min Wa',data[ data['FuzzPredict']==c]['WA'].max(),data[ data['FuzzPredict']==c]['WA'].min()
    print 'Max and min TLE',data[ data['FuzzPredict']==c]['TLE'].max(),data[ data['FuzzPredict']==c]['TLE'].min()
    print 'Submission user',data[ data['FuzzPredict'] == c]['submissions'].max(), data[ data['FuzzPredict'] == c]['submissions'].min()
    print 'Solved User user',data[ data['FuzzPredict'] == c]['solvedUser'].max(), data[ data['FuzzPredict'] == c]['solvedUser'].min()
    print data[ data['FuzzPredict']==c]['solvedUser'].mean()/data[ data['FuzzPredict']==c]['submissions'].mean()
    print '*'*40
'''print data['k_preds'].value_counts()
for c in range(4):
    print data[ data['k_preds']==c]['AC'].max(),data[ data['k_preds']==c]['AC'].min()
    print data[ data['k_preds']==c]['WA'].max(),data[ data['k_preds']==c]['WA'].min()
    print data[ data['k_preds']==c]['TLE'].max(),data[ data['k_preds']==c]['TLE'].min()
    print '*'*40
'''
'''
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
'''
