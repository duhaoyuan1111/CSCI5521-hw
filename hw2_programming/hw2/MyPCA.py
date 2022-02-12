import numpy as np

def PCA(X,num_dim=None):
    X_pca = X
    num_dim = num_dim
    mean = np.mean(X_pca, axis=0)
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    s = np.cov(np.transpose(X_pca-mean),ddof=0)
    #print(len(s)) # 784 * 784
    vals, vecs = np.linalg.eigh(s)
    #print(vals[0])
    #print(vals[1])
    #print(len(vals)) # 784
    #print(len(vecs)) # len(vecs[0]): 784  len(vecs): 784
    #for i in range(len(vals)):
    #    print(vals[i])
    totalVariance = sum(vals)
    per95Var = totalVariance * 0.95
    finalVec = []
    # select the reduced dimensions that keep >95% of the variance
    if num_dim is None:
        count = 0
        for i in range(783,-1,-1): # backwards => descending
            if vals[i] + count <= per95Var:
                count += vals[i]
                finalVec.append(vecs[:,i])
                #print("i:   ",i)
            else:
                count += vals[i]
                finalVec.append(vecs[:,i])
                #print("per95Var   ",per95Var)
                #print(count-vals[i])
                num_dim = 784-i
                break
        
    elif num_dim == 1:
        #print("dim == 1!")
        finalVec = vecs[:,783]
    # project the high-dimensional data to low-dimensional one
    finalVec = np.array(finalVec)        
    #print(len(finalVec)) # 128 rows
    #print(len(finalVec[0])) # 784 columns
    #print(len(X_pca))
    #print(len(X_pca[0]))
    X_pca = np.dot(X_pca-mean, np.transpose(finalVec)) # X_pca 3000*784

    if num_dim == 1:
        X_pca = np.transpose(np.array([X_pca]))
        #print(num_dim)
        #print(len(X_pca[0]))
    #print(num_dim)


    return X_pca, num_dim
