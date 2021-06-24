import numpy as np

def sigmoid(z: float) -> float:
    return 1.0 / (1 + np.exp(-z))

class Ctrnn(object):
    def __init__(self, size: int = 2):
        self.size = size
