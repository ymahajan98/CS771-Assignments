import numpy as np
#loading data
X_seen=np.load('X_seen.npy')
Xtest=np.load('Xtest.npy')
Ytest=np.load('Ytest.npy',)
class_attributes_seen=np.load('class_attributes_seen.npy')
class_attributes_unseen=np.load('class_attributes_unseen.npy')
means=np.zeros(shape=(50,4096))
for i in range(0,40):
    means[i]=X_seen[i].mean(axis=0)
mean_unseen=np.zeros(shape=(10,4096))

#computing similarity and unknown means
for i in range(0,10):
    similar=np.zeros(40)
    sum1=0
    for j in range (0,40):
        similar[j]=np.dot(class_attributes_seen[j],class_attributes_unseen[i])
    similar=similar/similar.sum()
    for j in range (0,40):
        mean_unseen[i]=mean_unseen[i]+similar[j]*means[j]
    means[i+40]=mean_unseen[i]
Y_output=np.zeros(6180)

#computing accuracy
ct=0
for i in range(0,6180):
    temp=np.zeros(4096)
    mini=np.zeros(10)
    for j in range(40,50):
        temp=Xtest[i]-means[j]
        mini[j-40]=np.sqrt(temp.dot(temp))
    Y_output[i]=1+np.argmin(mini)
    if Y_output[i]==Ytest[i]:
        ct=ct+1
accuracy=(ct*100.0)/6180
print(accuracy)
