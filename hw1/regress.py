import numpy as np
#loading data
X_seen=np.load('X_seen.npy')    
Xtest=np.load('Xtest.npy')
Ytest=np.load('Ytest.npy',) 
class_attributes_seen=np.load('class_attributes_seen.npy')  
class_attributes_unseen=np.load('class_attributes_unseen.npy') 
means=np.zeros(shape=(40,4096))
for i in range(0,40):
    means[i]=X_seen[i].mean(axis=0)

#l is lambda
l=100

#computing w and unknown means
identity=np.identity(85)
identity=l*identity
w=np.dot(np.transpose(class_attributes_seen),class_attributes_seen)+identity
w=np.linalg.inv(w)
w=np.dot(w,np.transpose(class_attributes_seen))
w=np.dot(w,means)
means_unseen=np.zeros(shape=(10,4096))
means_unseen=np.dot(class_attributes_unseen,w)

#calculating predictions
Y_output=np.zeros(6180)
ct=0
for i in range(0,6180):
    temp=np.zeros(4096)
    mini=np.zeros(10)
    for j in range(0,10):
        temp=Xtest[i]-means_unseen[j]
        mini[j]=np.sqrt(temp.dot(temp))
    Y_output[i]=1+np.argmin(mini)
    if Y_output[i]==Ytest[i]:
        ct=ct+1
gh=(ct*100.0)/6180
print(gh)
