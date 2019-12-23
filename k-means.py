import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()
K = 2 #number of clusters
epoches = 100 #number of iteration
eps = 1e-6
data = pd.read_csv('OldFaithful.csv')
print('data',data.keys().shape[0]) #number of attributes of dataset.
x = np.expand_dims(data['TimeEruption'].to_numpy(),axis=1)
y = np.expand_dims(data['TimeWaiting'].to_numpy(),axis=1)

dataset = np.hstack((x,y)) # dataset stacked horizontally

for i in range(dataset.shape[1]):
    dataset[:,i] = (dataset[:,i]-dataset[:,i].mean())/dataset[:,i].std()


mean = np.random.rand(K,dataset.shape[1])
score_map = np.zeros((dataset.shape[0],K))

for epoch in range(epoches):
    score_map = np.zeros((dataset.shape[0],K))
    for i in range(K):
        score_map[:,i] = np.sum(np.power(dataset-mean[i,:],2),axis=1)
    k_index = np.argmin(score_map,axis=1)

    for i in range(K):
        mean[i,:] = np.sum(dataset[np.where(k_index==i)[0],:],axis=0)/(np.where(k_index==i)[0].shape[0]+eps)

    for i in range(K):
        plt.scatter(dataset[np.where(k_index==i)[0],0],dataset[np.where(k_index==i)[0],1])
        plt.scatter(mean[:,0],mean[:,1],marker='+')
        plt.savefig('cluster.png')
        plt.pause(1e-5)
        