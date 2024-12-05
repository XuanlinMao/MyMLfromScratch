from modules import *
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def __init__(self):
        self.layers = []
    
    def __call__(self, x):
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass
