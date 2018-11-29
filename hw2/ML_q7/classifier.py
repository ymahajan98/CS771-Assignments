import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
#For the first dataset
Data_1=np.loadtxt('binclass.txt',dtype=float,delimiter=',')
X_1=Data_1[:,0]
Y_1=Data_1[:,1]
XY=Data_1[:,[0,1]]
#calculating mean
mp_1x=0; mp_1y=0; mi_1x=0; mi_1y=0; tp_1=0; ti_1=0; devp_1=0; devi_1=0;
for i in range(0,Data_1.shape[0]):
    if Data_1[i][2] == -1:
        tp_1=tp_1+1
        mp_1x=mp_1x+Data_1[i][0]
        mp_1y=mp_1y+Data_1[i][1]
    if Data_1[i][2] == 1:
        ti_1=ti_1+1
        mi_1x=mi_1x+Data_1[i][0]
        mi_1y=mi_1y+Data_1[i][1]
mp_1x=mp_1x/tp_1;mp_1y=mp_1y/tp_1;mi_1x=mi_1x/ti_1;mi_1y=mi_1y/ti_1;
#calculating standard deviation
for i in range(0,Data_1.shape[0]):
    if Data_1[i][2] == -1:
        devp_1=devp_1+(Data_1[i][0]-mp_1x)*(Data_1[i][0]-mp_1x)+(Data_1[i][1]-mp_1y)*(Data_1[i][1]-mp_1y)
    if Data_1[i][2] == 1:
        devi_1=devi_1+(Data_1[i][0]-mi_1x)*(Data_1[i][0]-mi_1x)+(Data_1[i][1]-mi_1y)*(Data_1[i][1]-mi_1y)
devp_1=devp_1/(2*tp_1)
devi_1=devi_1/(2*ti_1)
col=[]
#assigning colors
for i in range(0,len(X_1)):
    if Data_1[i][2]==-1:
        col.append('b')
    else:
        col.append('r')
#plotting quadratic boundary
plt.figure(0)
plt.scatter(X_1,Y_1,c=col,s=5)
x = np.linspace(-30., 30.)
y = np.linspace(-30., 30.)[:, None]
plt.contour(x, y.ravel(), devi_1*(x-mp_1x)**2+devi_1*(y-mp_1y)**2-devp_1*(x-mi_1x)**2-devp_1*(y-mi_1y)**2+devp_1*devi_1*math.log(devp_1/devi_1), [0])
plt.savefig('a1.png')
#plotting linear generative boundary
plt.figure(1)
plt.scatter(X_1,Y_1,c=col,s=5)
x = np.linspace(-30., 30.)
y = np.linspace(-30., 30.)[:, None]
plt.contour(x, y.ravel(), (x-mp_1x)**2+(y-mp_1y)**2-(x-mi_1x)**2-(y-mi_1y)**2, [0])
plt.savefig('a2.png')
#plotting boundary using SVM
clf= svm.SVC(kernel='linear', C=1)
clf.fit(XY,col)
w=clf.coef_[0]
w1=-w[0]/w[1]
plt.figure(2);
plt.scatter(X_1,Y_1,c=col,s=5)
x = np.linspace(-30., 30.)
y = w1*x-clf.intercept_[0]/w[1]
plt.plot(x,y)
plt.savefig('a3.png')

#For the second dataset,same code as previous part
Data=np.loadtxt('binclassv2.txt',dtype=float,delimiter=',')
X_2=Data[:,0]
Y_2=Data[:,1]
XY=Data[:,[0,1]]
mp_1x=0; mp_1y=0; mi_1x=0; mi_1y=0; tp_1=0; ti_1=0; devp_1=0; devi_1=0;
for i in range(0,Data.shape[0]):
    if Data[i][2] == -1:
        tp_1=tp_1+1
        mp_1x=mp_1x+Data[i][0]
        mp_1y=mp_1y+Data[i][1]
    if Data[i][2] == 1:
        ti_1=ti_1+1
        mi_1x=mi_1x+Data[i][0]
        mi_1y=mi_1y+Data[i][1]
mp_1x=mp_1x/tp_1;mp_1y=mp_1y/tp_1;mi_1x=mi_1x/ti_1;mi_1y=mi_1y/ti_1;
for i in range(0,Data.shape[0]):
    if Data[i][2] == -1:
        devp_1=devp_1+(Data[i][0]-mp_1x)*(Data[i][0]-mp_1x)+(Data[i][1]-mp_1y)*(Data[i][1]-mp_1y)
    if Data[i][2] == 1:
        devi_1=devi_1+(Data[i][0]-mi_1x)*(Data[i][0]-mi_1x)+(Data[i][1]-mi_1y)*(Data[i][1]-mi_1y)
devp_1=devp_1/(2*tp_1)
devi_1=devi_1/(2*ti_1)
col=[]
for i in range(0,len(X_2)):
    if Data[i][2]==-1:
        col.append('b')
    else:
        col.append('r')
plt.figure(4)
plt.scatter(X_2,Y_2,c=col,s=5)
x = np.linspace(-40., 40.)
y = np.linspace(-40., 40.)[:, None]
plt.contour(x, y.ravel(), devi_1*(x-mp_1x)**2+devi_1*(y-mp_1y)**2-devp_1*(x-mi_1x)**2-devp_1*(y-mi_1y)**2+devp_1*devi_1*math.log(devp_1/devi_1), [0])
plt.savefig('a4.png')
plt.figure(5)
plt.scatter(X_2,Y_2,c=col,s=5)
x = np.linspace(-40., 40.)
y = np.linspace(-40., 40.)[:, None]
plt.contour(x, y.ravel(), (x-mp_1x)**2+(y-mp_1y)**2-(x-mi_1x)**2-(y-mi_1y)**2, [0])
plt.savefig('a5.png')
clf= svm.SVC(kernel='linear', C=1)
clf.fit(XY,col)
w=clf.coef_[0]
w1=-w[0]/w[1]
plt.figure(6);
plt.scatter(X_2,Y_2,c=col,s=5)
x = np.linspace(-30., 30.)
y = w1*x-clf.intercept_[0]/w[1]
plt.plot(x,y)
plt.savefig('a6.png')
