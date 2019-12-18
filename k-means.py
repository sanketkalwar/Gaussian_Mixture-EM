import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K = 3 #number of clusters
data = pd.read_csv('OldFaithful.csv')
print('data',data.keys().shape[0]) #number of attributes of dataset.
x = np.expand_dims(data['TimeEruption'].to_numpy(),axis=1)
y = np.expand_dims(data['TimeWaiting'].to_numpy(),axis=1)

dataset = np.hstack((x,y)) # dataset stacked horizontally

for i in range(dataset.shape[1]):
    dataset[:,i] = (dataset[:,i]-dataset[:,i].mean())/dataset[:,i].std()
    
mean = np.random.rand(K,dataset.shape[1])
score_map = np.zeros((dataset.shape[0],K))
print('Number of clusters:-',K)
print('dataset',dataset.shape)
print('Mean shape',mean.shape)
print('Score map',score_map.shape)

for n in range(dataset.shape[0]):
    list_temp = []
    for k in range(K):
        list_temp.append(np.sum(np.power(dataset[n,:]-mean[k,:],2)))
    score_map[n,np.array(list_temp).argmax()] = 1

print('score_map',score_map)
