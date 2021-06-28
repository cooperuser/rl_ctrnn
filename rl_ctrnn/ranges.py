import numpy as np

class Range(object):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def set_clamp(self, value: float):
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def __repr__(self):
        return f"Range[{self.min:0.4f} {self.max:0.4f}]"

class CtrnnRanges(object):
    def __init__(self):
        self.weights = Range(-16, 16)
        self.biases = Range(-16, 16)
        self.time_constants = Range(0.5, 10)

    def set_weight_range(self, min=None, max=None):
        self.weights.min = min or self.weights.min
        self.weights.max = max or self.weights.max

    def set_bias_range(self, min=None, max=None):
        self.biases.min = min or self.biases.min
        self.biases.max = max or self.biases.max

    def set_time_constant_range(self, min=None, max=None):
        self.time_constants.min = min or self.time_constants.min
        self.time_constants.max = max or self.time_constants.max

    def get_weights(self, neurons: int = 2) -> np.ndarray:
        return np.random.uniform(
            self.weights.min,
            self.weights.max,
            size=(neurons, neurons)
        )

    def get_biases(self, neurons: int = 2) -> np.ndarray:
        return np.random.uniform(
            self.biases.min,
            self.biases.max,
            size=neurons
        )

    def get_time_constants(self, neurons: int = 2) -> np.ndarray:
        return np.random.uniform(
            self.time_constants.min,
            self.time_constants.max,
            size=neurons
        )
