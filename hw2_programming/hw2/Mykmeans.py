#import libraries
import numpy as np
import math

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005] # indices for the samples
        sample_center = []

        num_iter = 0 # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False
        # iteratively update the centers of clusters till convergence
        while not is_converged:
            
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)): #0 ~ 2999
                # use euclidean distance to measure the distance between sample and cluster centers
                findMin = []
                for k in range(8):
                    if num_iter == 0:
                        findMin.append(np.linalg.norm(X[i]-X[init_idx[k]],ord=2))
                    else:
                        findMin.append(np.linalg.norm(X[i]-self.center[k],ord=2))
                bestMeanIndex = findMin.index(min(findMin))
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = bestMeanIndex # cluster 0 - 7

            # update the centers based on cluster assignment (M step)
            FindMin = []
            for d in range(8): # 8
                sum = []
                
                for v in range(3000):
                    if num_iter == 0:
                        if init_idx[cluster_assignment[v]] == init_idx[d]:
                            sum.append(X[v])
                    else:
                        if cluster_assignment[v] == d:
                            sum.append(X[v])
                meanVector = np.mean(sum,axis=0) # 784
                
                FindMin.append(meanVector)

            self.center = FindMin
            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)
            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1
        # compute the information entropy for different clusters
        entropyList = []
        for q in range(8):
            totalNumCluster = 0
            subEntropy = 0
            c_0 = 0
            c_8 = 0
            c_9 = 0
            for w in range(3000):
                if cluster_assignment[w] == q:
                    totalNumCluster += 1
                    if y[w] == 0:
                        c_0 += 1
                    elif y[w] == 8:
                        c_8 += 1
                    else:
                        c_9 +=1
            if c_0 != 0:
                subEntropy += c_0/totalNumCluster*math.log2(c_0/totalNumCluster)
            if c_8 != 0:
                subEntropy += c_8/totalNumCluster*math.log2(c_8/totalNumCluster)
            if c_9 != 0:
                subEntropy += c_9/totalNumCluster*math.log2(c_9/totalNumCluster)
            subEntropy = subEntropy * (-1)
            entropyList.append(subEntropy)
        
        enSum = 0
        for g in range(8):
            enSum += entropyList[g]
        enSum = enSum/8
        entropy = enSum

        return num_iter, self.error_history, entropy

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        errorSum = 0
        for i in range(3000):
            errorSum += pow(np.linalg.norm(X[i] - self.center[cluster_assignment[i]],ord=2),2)
        error = errorSum
        return error

    def params(self):
        return self.center
