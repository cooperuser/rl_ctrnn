from typing_extensions import TypeAlias
import numpy as np

Array: TypeAlias = np.ndarray

def sigmoid(z: Array) -> Array:
    return 1.0 / (1 + np.exp(-z))

class Range(object):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def set_clamp(self, value: float):
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def __repr__(self):
        return f"R<{self.min} {self.max}>"

class Ctrnn(object):
    def __init__(self, size: int = 2):
        self.size = size
        self.reset()

    def reset(self):
        self.biases = np.zeros(self.size)
        self.time_constants = np.ones(self.size)
        self._inv_time_constants = 1.0 / self.time_constants
        self.weights = np.zeros((self.size, self.size))

    def make_instance(self) -> Array:
        return np.zeros(self.size)

    def set_bias(self, neuron: int, bias: float):
        self.biases[neuron] = bias

    def set_time_constant(self, neuron: int, time_constant: float):
        self.time_constants[neuron] = time_constant
        self._inv_time_constants[neuron] = 1.0 / self.time_constants[neuron]

    def set_weight(self, pre: int, post: int, weight: float):
        self.weights[pre][post] = weight

    def randomize(self, weights: Range, biases: Range, t_c: Range, seed=None):
        if seed != None:
            np.random.seed(seed)
        s = self.size
        self.weights = np.random.uniform(weights.min, weights.max, size=(s, s))
        self.biases = np.random.uniform(biases.min, biases.max, size=s)
        self.time_constants = np.random.uniform(t_c.min, t_c.max, size=s)

    def step(self, dt: float, voltages: Array, inputs: Array=None) -> Array:
        inputs = inputs if inputs != None else np.zeros(self.size)
        net = inputs + np.dot(self.weights.T, sigmoid(voltages + self.biases))
        return voltages + dt * (self._inv_time_constants * (-voltages + net))

    def get_output(self, voltages: Array):
        return sigmoid(voltages + self.biases)

if __name__ == "__main__":
    pass
