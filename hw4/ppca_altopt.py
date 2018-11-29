import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#data=np.load("facedata.pkl");
data=scipy.io.loadmat('../facedata.mat')
V=data['X']
X=V.astype('float')

'''
for i in range(0,5):
	a=X[i:i+1,:]
	b=np.resize(a,(64,64))
	b=b.T
	plt.imshow(b,cmap="gray")
	plt.show()
'''
d=4096
mat=[100]
for k in mat:
	
	n=X.shape[0]
	Z=np.random.rand(n,k)
	w=np.random.rand(d,k)
	mean=X.mean(0)
	for j in range(0,X.shape[0]):
		X[j]=np.subtract(X[j],mean)

	for j in range(0,100):
		w=np.matmul(np.matmul(np.matrix(X.T),np.matrix(Z)),np.linalg.inv(np.matmul(np.matrix(Z.T),np.matrix(Z))))
		Z=np.matmul(np.matmul(np.matrix(X),np.matrix(w)),np.linalg.inv(np.matmul(np.matrix(w.T),np.matrix(w))))

	recon=np.matmul(Z,w.T)
	for j in range(0,recon.shape[0]):
		recon[j]=recon[j]+mean
print X
print recon
	#Compare Original and reonstructed
'''	for i in range(0,5):
		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
		a=recon[i:i+1,:]
		b=np.resize(a,(64,64))
		b=b.T
		ax1.imshow(b,cmap="gray")
		a=X[i:i+1,:]
		b=np.resize(a,(64,64))
		b=b.T
		ax2.imshow(b,cmap="gray")
		ax1.set_xlabel('Regenerated')
		ax2.set_xlabel('Original')
		ax1.set_title("K is"+str(k))
		plt.show()
		#plt.savefig("k"+str(k)+"-img-comparison-"+str(i)+".png")
		
	w=w.T
	for i in range(0,10):
		a=w[i:i+1,:]
		b=np.resize(a,(64,64))
		b=b.T
		plt.imshow(b,cmap="gray")	
		plt.show()	
		#plt.savefig("k"+str(k)+"-basis-img-0"+str(i)+".png")'''
