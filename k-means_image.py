import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
plt.ion()
K = 4 #number of clusters
epoches = 10 #number of iteration
eps = 1e-6
data = cv2.imread('test_img.jpeg')
h,w,c = data.shape[0],data.shape[1],data.shape[2]
dataset = data.reshape((h*w,c))
dataset = (dataset-dataset.mean())/dataset.std()

for i in range(dataset.shape[1]):
    dataset[:,i] = (dataset[:,i]-dataset[:,i].mean())/dataset[:,i].std()
    
mean = np.random.rand(K,dataset.shape[1])
score_map = np.zeros((dataset.shape[0],K))

l = {0:np.array([255,0,0]),1:np.array([0,255,0]),2:np.array([0,0,255]),3:np.array([0,255,255]),4:np.array([50,50,255])}

for epoch in range(epoches):
    img_temp = np.zeros((h*w,3))
    score_map = np.zeros((dataset.shape[0],K))
    for i in range(K):
        score_map[:,i] = np.sum(np.power(dataset-mean[i,:],2),axis=1)
    k_index = np.argmin(score_map,axis=1)
    for i in range(K):
        mean[i,:] = np.sum(dataset[np.where(k_index==i)[0],:],axis=0)/(np.where(k_index==i)[0].shape[0]+eps)
        img_temp[np.where(k_index==i)[0],:] = l[i]
    plt.imshow(img_temp.reshape((h,w,c)))
    plt.savefig('out_img.png')
    plt.pause(1e-6)
