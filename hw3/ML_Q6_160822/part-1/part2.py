import numpy as np
import math
import random
import matplotlib.pyplot as plt

data_train=np.genfromtxt('ridgetrain.txt')
data_test=np.genfromtxt('ridgetest.txt')
gamma=0.1
lamda=0.1
L=100
X_train=data_train[:,0]
Y_train=data_train[:,1]
X_test=data_test[:,0]
Y_test=data_test[:,1]
landmarks=random.sample(set(X_train),L)
psi=np.zeros((X_train.size,L))
psi_test=np.zeros((X_test.size,L))
for i in range (0,X_train.size):
    for j in range(0,L):
        psi[i][j]=math.exp(-gamma*(X_train[i]-landmarks[j])**2)
for i in range (0,X_test.size):
    for j in range(0,L):
        psi_test[i][j]=math.exp(-gamma*(X_test[i]-landmarks[j])**2)
I=np.identity(X_train.size)
k=np.zeros((X_train.size,X_train.size))
I=lamda*I
for i in range (0,X_train.size):
    for j in range(0,X_train.size):
        k[i][j]=psi[i].dot(np.transpose(psi[j]))
k=np.linalg.inv(k+I)
k=k.dot(Y_train)
Y_pred=np.zeros(Y_test.size)
for i in range(0,Y_pred.size):
    Y_pred[i]=0
    for j in range(0,k.size):
        Y_pred[i]=Y_pred[i]+k[j]*psi_test[i].dot(np.transpose(psi[j]))
rmse=0
for i in range(0,Y_pred.size):
    rmse=rmse+(Y_pred[i]-Y_test[i])**2
rmse=rmse/ Y_pred.size
rmse=math.sqrt(rmse)
print (rmse)
plt.figure(0)
plt.scatter(X_train,Y_train,c='blue')
plt.scatter(X_test,Y_pred,c='red')
plt.savefig('part-2(L=100).png')
plt.show()
