import numpy as np

class KNN:
    def __init__(self, k:int) -> None:
        self.k = k
    
    def predict(self, X:np.ndarray, Xtrain:np.ndarray, ytrain:np.ndarray) -> np.ndarray:
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            idx = ((Xtrain-X[i,:])**2).sum(axis=1).argsort()[:self.k]
            v,c = np.unique(ytrain[idx], return_counts=True)
            pred[i] = v[c.argmax()]
        return pred

