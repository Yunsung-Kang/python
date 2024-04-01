# packages
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# iris data
iris = sns.load_dataset('iris')
input_var = iris.iloc[:,[0,1,2,3]]
target_var = iris.iloc[:,4]

# Kmeans
kmeans = KMeans(n_clusters= 3,
                init= 'k-means++',
                max_iter= 300,
                random_state= 0)
kmeans.fit(input_var)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

input_var['target'] = target_var
input_var['cluster'] = kmeans.labels_
iris_result = input_var.groupby(['target','cluster'])['sepal_length'].count()
print(iris_result)

# silhouette
iris1 = sns.load_dataset('iris')

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
y= iris1.iloc[:,4]
y= encoder.fit_transform(y)
print(y)
iris1['species'] = y
iris_pd = pd.DataFrame(iris1.iloc[:,0:4])

kmeans1 = KMeans(n_clusters=3, init='k-means++', max_iter=300,random_state=1234).fit(iris_pd)
iris_pd['cluster'] = kmeans1.labels_
print(iris_pd)
iris_pd['cluster']

score_samples = silhouette_samples(iris1, iris_pd['cluster'])
print(score_samples.shape)
print(score_samples[0:10])

iris_pd['silhouette_coeff']= score_samples
average_score= silhouette_score(iris1,iris_pd['cluster'])
print('Silhouette Coefficiend= {0:3f}'.format(average_score))
print(iris_pd.groupby('cluster')['silhouette_coeff'].mean())