#encoding:utf8
import numpy as np
import random
#from scipy import stats
#from sklearn.cluster import KMeans
from sklearn import datasets
#from sklearn.metrics import euclidean_distances
#import codecs
class KMeans:
    def __init__(self, k = 2):
        #k是簇的个数
        self.k = k
        #self.kmeans_args = kmeans_args
        self._data = None
        self._labels = None
        self._centers = None
        #self.labels = None

    def fit(self, X):
        '''
        X : 数据集，矩阵形式的二维数组
        '''
        self._data = X
        #随机在X中选择k个样本做簇的中心
        clusters = np.array(random.sample(X, self.k))
        self.labels = np.array([-1]*len(X))

        #对所有的样本计算与各簇中心的距离，选择最近的簇加入
        for idx in range(len(X)):
            distances = np.array([sum((X[idx]-cluster)*(X[idx]-cluster)) for cluster in clusters])
            self.labels[idx] = np.argmin(distances)

        isUpdated = True
        cnt = 1
        #对k个簇重新选定簇中心，并重新分配样本
        while isUpdated:
            print cnt
            cnt += 1
            isUpdated, clusters = self.update(clusters)
            for idx in range(len(X)):
                distances = np.array([sum((X[idx]-cluster)*(X[idx]-cluster)) for cluster in clusters])
                self.labels[idx] = np.argmin(distances)
        self._labels = self.labels
        self._centers = clusters
        return self

    #把簇的均值作为新的簇中心
    def update(self, clusters):
        old = clusters.copy()
        for i in range(self.k):
            clusters[i] = self._data[self.labels==i].mean()
        delta = [sum((old[i]-clusters[i])*(old[i]-clusters[i])) for i in range(self.k)]
        if sum(delta) < 1e-6:
            return False, clusters
        else:
            return True, clusters

    #把x与所有的簇中心计算距离，加入最近的簇
    def predict(self, x):
        #for idx in range(self.k):
        x = np.array(x)
        distances = np.array([sum((self._centers[idx]-x)*(self._centers[idx]-x)) for idx in range(self.k)])
        return np.argmin(distances)

iris = datasets.load_iris()
kmeans = KMeans(k=3)
kmeans.fit(iris['data'])
print [kmeans.predict(x) for x in iris['data']]
