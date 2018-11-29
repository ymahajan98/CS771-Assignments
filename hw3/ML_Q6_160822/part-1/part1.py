import numpy as np
import math
import matplotlib.pyplot as plt

data_train=np.genfromtxt('ridgetrain.txt')
data_test=np.genfromtxt('ridgetest.txt')
gamma=0.1
lamda=100
X_train=data_train[:,0]
Y_train=data_train[:,1]
X_test=data_test[:,0]
Y_test=data_test[:,1]
I=np.identity(X_train.size)
k=np.zeros((X_train.size,X_train.size))
I=lamda*I
for i in range (0,X_train.size):
    for j in range(0,X_train.size):
        k[i][j]=math.exp(-gamma*(X_train[i]-X_train[j])**2)
k=np.linalg.inv(k+I)
k=k.dot(Y_train)
Y_pred=np.zeros(Y_test.size)
for i in range(0,Y_pred.size):
    Y_pred[i]=0
    for j in range(0,k.size):
        Y_pred[i]=Y_pred[i]+k[j]*math.exp(-gamma*(X_test[i]-X_train[j])**2)
rmse=0
for i in range(0,Y_pred.size):
    rmse=rmse+(Y_pred[i]-Y_test[i])**2
rmse=rmse/ Y_pred.size
rmse=math.sqrt(rmse)
print (rmse)
plt.figure(0)
plt.scatter(X_train,Y_train,c='blue')
plt.scatter(X_test,Y_pred,c='red')
plt.savefig('part-1(lambda=100).png')
plt.show()
