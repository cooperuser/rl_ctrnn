from typing_extensions import TypeAlias

from rl_ctrnn.ctrnn import Ctrnn
import numpy as np
import numpy.typing as npt

Array: TypeAlias = np.ndarray
Datum: TypeAlias = npt.NDArray[np.float32]


class Clamp(object):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max


class Bounds(object):
    def __init__(self) -> None:
        self.flux = Clamp(0, 10)
        self.center = Clamp(-16, 16)
        self.period = Clamp(2, 10)


class RLCtrnn(object):
    def __init__(self, ctrnn: Ctrnn, seed):
        self.ctrnn = ctrnn
        self.bounds = Bounds()
        self.rng = np.random.default_rng(seed)

        shape = (ctrnn.size, ctrnn.size)
        self.flux: float = 1
        self.center = self.ctrnn.weights
        self.period = np.random.uniform(
            self.bounds.period.min, self.bounds.period.max, size=shape
        )
        self.time: Datum = np.zeros(shape)

        self.flux_conv_rate = 0.1
        self.max_flux_amp = 10
        self.learn_rate = 1.0
        self.distance = 0.0

    def update(self, reward: float = 0):
        """Tweak the weights of the network using a reward value"""
        self.flux -= self.flux_conv_rate * self.max_flux_amp * reward
        self.flux = np.clip(self.flux, 0, self.max_flux_amp)
        displacement = self.flux * np.sin(2 * np.pi * self.time / self.period)
        center = self.center + self.learn_rate * displacement * reward
        center = center.clip(self.bounds.center.min, self.bounds.center.max)
        self.distance += np.sqrt(np.sum(np.power(center + -self.center, 2)))
        self.center = center

    def step(self, voltages: Array = np.empty(0), dt: float = 0.05):
        """Wrapper around normal Ctrnn.step method that sets weights using centers, periods, and flux"""
        size = self.ctrnn.size
        self.time += dt
        time = self.time.flat
        period = self.period.flat
        stale = filter(lambda i: time[i] > period[i], range(size ** 2))
        for i in stale:
            self.time.put(i, 0)
            p = self.bounds.period
            self.period.put(i, self.rng.uniform(p.min, p.max))
        displacement = self.flux * np.sin(2 * np.pi * self.time / self.period)
        self.ctrnn.weights = self.center + displacement
        return self.ctrnn.step(dt, voltages)
