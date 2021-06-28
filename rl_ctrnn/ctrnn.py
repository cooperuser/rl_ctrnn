from typing_extensions import TypeAlias
import numpy as np

from .ranges import Range, CtrnnRanges

Array: TypeAlias = np.ndarray

def sigmoid(z: Array) -> Array:
    return 1.0 / (1 + np.exp(-z))

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

    def randomize(self, ranges: CtrnnRanges, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = ranges.get_weights(self.size)
        self.biases = ranges.get_biases(self.size)
        self.time_constants = ranges.get_time_constants(self.size)
        self._inv_time_constants = 1.0 / self.time_constants

    @DeprecationWarning
    def randomize_custom(
        self,
        weights: Range = None,
        biases: Range = None,
        time_constants: Range = None,
        seed=None
    ):
        if seed != None:
            np.random.seed(seed)
        s = self.size
        if weights != None:
            self.weights = np.random.uniform(
                weights.min, weights.max, size=(s, s))
        if biases != None:
            self.biases = np.random.uniform(
                biases.min, biases.max, size=s)
        if time_constants != None:
            self.time_constants = np.random.uniform(
                time_constants.min, time_constants.max, size=s)
            self._inv_time_constants = 1.0 / self.time_constants

    def step(self, dt: float, voltages: Array, inputs: Array=None) -> Array:
        inputs = inputs if inputs != None else np.zeros(self.size)
        net = inputs + np.dot(self.weights.T, sigmoid(voltages + self.biases))
        return voltages + dt * (self._inv_time_constants * (-voltages + net))

    def get_output(self, voltages: Array):
        return sigmoid(voltages + self.biases)

if __name__ == "__main__":
    pass
