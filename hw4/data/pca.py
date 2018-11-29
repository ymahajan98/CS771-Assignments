import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = sio.loadmat("mnist_small.mat", mdict=None)

pca = PCA(n_components=2)
X_small =  pca.fit_transform(data['X'])
for i in range(10):
	kmeans = KMeans(n_clusters=10, init='random')
	cluster = kmeans.fit_predict(X_small)

	plt.scatter(X_small[:,0], X_small[:,1], c=cluster,s=5)
	plt.savefig("PCA/PCA_number"+str(i+1))
	plt.close()

