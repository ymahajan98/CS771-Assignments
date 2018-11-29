import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random

data=scipy.io.loadmat('facedata.mat')
data=data['X'].astype(float)
dimension=64*64
rows=165
iterations=100
k=100
Z=np.random.rand(rows,k)
W=np.random.rand(dimension,k)
data1=data
mean1=data.mean(axis=0)
for i in range(0,165):
    data[i]=data[i]-data.mean(axis=0)
for i in range(0,iterations):
    W=(np.transpose(data)).dot(Z)
    W=W.dot(np.linalg.inv((np.transpose(Z)).dot(Z)))
    Z=data.dot(W)
    Z=Z.dot(np.linalg.inv((np.transpose(W)).dot(W)))
W_T=np.transpose(W)
images_number=[10,20,30,40,50]
new_X=Z.dot(np.transpose(W))
for i in range(0,165):
    new_X[i]=new_X[i]+mean1
j=1
for i in images_number:
    image1=np.transpose(np.reshape(data1[i],(64,64)))
    plt.imshow(image1,cmap='gray')
    plt.savefig("k="+str(k)+",original"+str(j)+".png")
    plt.close()

    image2=np.transpose(np.reshape(new_X[i],(64,64)))
    plt.imshow(image2,cmap='gray')
    plt.savefig("k="+str(k)+",recreated"+str(j)+".png")
    plt.close()
    j=j+1
j=1
for i in range(0,10):
    image1=np.transpose(np.reshape(W_T[i],(64,64)))
    plt.imshow(image1,cmap='gray')
    plt.savefig("k="+str(k)+",basis"+str(j)+".png")
    plt.close()
    j=j+1

