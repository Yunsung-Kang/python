# packages
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# iris data
iris = sns.load_dataset('iris')
input_var = iris.iloc[:,[0,1,2,3]]
target_var = iris.iloc[:,4]

sns.scatterplot(x= 'sepal_length',
                y= 'sepal_width',
                hue= 'species',
                style= 'species',
                s=100,
                data=iris)
plt.show()

# StandardScaler
ss = StandardScaler()
ss.fit(input_var)
input_scaled = ss.transform(input_var)
print(input_scaled[0:5,])

# PCA
pca = PCA(n_components=2)
pca.fit(input_scaled)
iris_pca = pca.transform(input_scaled)
print(iris_pca.shape)
print(iris_pca[0:6,])

iris_pca = pd.DataFrame(iris_pca,
                        columns = ['pc1','pc2'])
print(iris_pca)

sns.scatterplot(x= 'pc1',
                y= 'pc2',
                hue= target_var,
                style= target_var,
                s=100,
                data=iris_pca)
plt.show()

print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))