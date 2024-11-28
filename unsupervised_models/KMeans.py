import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K) -> None:
        self.K = K
        self.MAXITER = 1000

    def _init_centroid(self, X: np.ndarray) -> np.ndarray:
        idx = np.random.choice(range(X.shape[0]), size=self.K, replace=False)
        return X[idx,:]

    def _update_centroid(self, X: np.ndarray) -> None:
        self.centroid = np.concatenate(
            [X[self.label==i,:].mean(axis=0).reshape(1,-1) for i in range(self.K)]
            , axis=0
        )

    def train(self, X: np.ndarray) -> None:
        self.n, self.p = X.shape
        self.centroid = self._init_centroid(X)
        it = 0
        while it < self.MAXITER:
            dist = np.concatenate(
                [((X-self.centroid[i,:])**2).mean(axis=1).reshape(-1,1) for i in range(self.K)]
                , axis=1)
            self.label = dist.argmin(axis=1)
            self._update_centroid(X)
            it += 1
        

    def predict(self) -> np.ndarray:
        return self.label
    

if __name__ == "__main__":
    np.random.seed(132)
    X = np.random.randn(500,2)
    kmeans = KMeans(4)
    kmeans.train(X)
    for i in range(len(np.unique(kmeans.predict()))):
        points = X[kmeans.predict()==i,:]
        plt.scatter(points[:,0], points[:,1])
    plt.title("result of Kmeans")
    plt.show()
