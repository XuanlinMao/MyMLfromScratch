from .modules import *
from .BaseModel import BaseModel

class MLP(BaseModel):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return super().forward(x)
    
    def backward(self, x):
        return super().backward(x)