import numpy as np
import math
import random
import matplotlib.pyplot as plt

data=np.genfromtxt('kmeans_data.txt')
X_data=data[:,0]
Y_data=data[:,1]
ans=np.zeros(X_data.size)
gamma=0.1
randnum=random.randint(0,X_data.size)
landmarks=data[randnum]
new_features=np.zeros(X_data.size)
for i in range(0,X_data.size):
    new_features[i]=math.exp(-gamma*(np.linalg.norm(data[i]-landmarks))**2)
mean_1=new_features[0]
mean_2=new_features[1]
while 1:
    new_mean1=0
    new_mean2=0
    ct1=0
    ct2=0
    for i in range(0,X_data.size):
        bc1=new_features[i]
        bc2=mean_1
        bc3=mean_2
        if abs(bc1-bc2) < abs(bc1-bc3):
            ans[i]=1
            ct1=ct1+1
        else:
            ans[i]=2
            ct2=ct2+1
    for i in range(0,X_data.size):
       if ans[i]==1:
           new_mean1=new_mean1+new_features[i]
       else:
           new_mean2=new_mean2+new_features[i]
    new_mean1=new_mean1/ct1
    new_mean2=new_mean2/ct2
    if new_mean1==mean_1 and new_mean2==mean_2:
        break
    else:
        mean_1=new_mean1
        mean_2=new_mean2
col=[]
for i in range(0,X_data.size):
    if i==randnum:
        col.append('blue')
        continue
    if ans[i]==1:
        col.append('red')
    else:
        col.append('green')
plt.figure(0)
plt.scatter(X_data,Y_data,c=col)
plt.scatter(landmarks[0],landmarks[1],c='blue')
plt.show()
