#import libraries
import numpy as np
from matplotlib import pyplot as plt
from Mykmeans import Kmeans
from MyPCA import PCA
import time

# read in data.
data=np.genfromtxt("Digits089.csv",delimiter=",")
X = data[:,2:]
y = data[:,1]

# apply kmeans algorithms to raw data
clf = Kmeans(k=8)
start = time.time()
num_iter, error_history, entropy_raw = clf.run_kmeans(X, y)
time_raw = time.time() - start

# plot the history of reconstruction error
fig = plt.figure()
plt.plot(np.arange(len(error_history)),error_history,'b-',linewidth=2)
fig.set_size_inches(10, 10)
fig.savefig('raw_data.png')
plt.show()

# apply kmeans algorithms to low-dimensional data (PCA) that captures >95% of variance
X_pca, num_dim = PCA(X)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca, error_history_pca, entropy_pca = clf.run_kmeans(X_pca, y)
time_pca = time.time() - start

# plot the history of reconstruction error
fig1 = plt.figure()
plt.plot(np.arange(len(error_history_pca)),error_history_pca,'b-',linewidth=2)
fig1.set_size_inches(10, 10)
fig1.savefig('pca.png')
plt.show()

# apply kmeans algorithms to 1D feature obtained from PCA
X_pca, _ = PCA(X,1)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca_1, error_history_pca_1, entropy_pca_1 = clf.run_kmeans(X_pca, y)
time_pca_1 = time.time() - start

# plot the history of reconstruction error
fig2 = plt.figure()
plt.plot(np.arange(len(error_history_pca_1)),error_history_pca_1,'b-',linewidth=2)
fig2.set_size_inches(10, 10)
fig2.savefig('pca_1d.png')
plt.show()

# print the average information entropy and number of iterations for convergence
print('Using raw data converged in %d iteration (%.2f seconds)' %(num_iter,time_raw))
print('Information entropy: %.3f' %entropy_raw)

print('#################')
print('Project data into %d dimensions with PCA converged in %d iteration (%.2f seconds)'%(num_dim,num_iter_pca,time_pca))
print('Information entropy: %.3f' %entropy_pca)

print('#################')
print('Project data into 1 dimension with PCA converged in %d iteration (%.2f seconds)'%(num_iter_pca_1,time_pca_1))
print('Information entropy: %.3f' %entropy_pca_1)
