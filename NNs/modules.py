import numpy as np
from typing import List
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def optimize(self, lr: float) -> None:
        pass




class Linear(Layer):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        self.weight = np.random.randn(dim_in, dim_out)
        self.bias = np.random.randn(1, dim_out)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x


    def backword(self, grad: np.ndarray) -> np.ndarray:
        """
        grad: n x n_out
        return: n x n_in
        """
        # record the grad of parameters
        self.d_weight = self.input.T @ grad
        self.d_bias = np.sum(grad, axis=0, keepdims=True)
        # return the grad of input
        return grad @ self.weight.T
    
    def optimize(self, lr: float) -> None:
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    

class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.zeros_like(x)
        mask = x > 0
        output[mask] = 1 / (1 + np.exp(-x))
        output[~mask] = np.exp(x) / (1 + np.exp(x))
        self.output = output
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.output * (1 - self.output)
    
    def optimize(self, lr: float) -> None:
        pass
    

    
class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0., x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.input > 0)
    
    def optimize(self, lr: float) -> None:
        pass