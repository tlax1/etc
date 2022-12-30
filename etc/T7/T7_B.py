import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]
print(X)

Y =pd.DataFrame(iris.target)
Y.columns = ["Targets"]
print(Y)

# KMeans
KM_model = KMeans(n_clusters=3)
KM_model.fit(X)

# EM for GMM
scalar =  preprocessing.StandardScaler()
scalar.fit(X)
xsa = scalar.transform(X)
xs = pd.DataFrame(xsa, colormap = X.columns)

# GMM
gmm = GaussianMixture(n_components=40)
gmm.fit(X)

colormap = np.array(['red', 'blue', 'green'])

plt.figure(figsize=(14,7))

plt.subplot(1,2,1)
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[Y.Targets], s=40)
plt.title("Sepal")
plt.xlabel("Sepal_Length")
plt.ylabel("Sepal_Width")

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title("Petal")
plt.xlabel("Petal_Length")
plt.ylabel("Petal_Width")
plt.show()


plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[KM_model.labels_], s=40)
plt.title("KMeans")
plt.xlabel("Petal_Length")
plt.ylabel("Petal_Width")

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm.predict(X)], s=40)
plt.title("KMeans")
plt.xlabel("Petal_Length")
plt.ylabel("Petal_Width")
plt.show