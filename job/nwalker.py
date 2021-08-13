from random import sample
from typing_extensions import TypeAlias
from behavior.oscillator import Oscillator
from typing import List, Tuple
from rl_ctrnn.ctrnn import Ctrnn
import numpy as np

Pair: TypeAlias = Tuple[Ctrnn, float]


class NWalker(object):
    def __init__(
        self,
        ctrnn: Ctrnn,
        seed: int = 0,
        mutation: float = 0.05,
        duration: float = 10,
        samples: int = 1,
        dt: float = 0.01,
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.mutation = mutation
        self.attempts: List[Pair] = []
        self.attempt = 0
        self.best = 0
        self.dt = dt
        self.duration = duration
        self.samples = samples
        self.rng = np.random.default_rng(self.seed)

    def new_behavior(self, state: np.ndarray) -> Oscillator:
        b = Oscillator(self.dt, duration=self.duration, window=self.duration / 6)
        b.setup(state)
        return b

    def setup(self):
        voltages = self.progenitor.make_instance()
        behavior = self.new_behavior(self.progenitor.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = self.progenitor.step(self.dt, voltages)
            behavior.grade(self.progenitor.get_output(voltages))
        self.attempts.append((self.progenitor, behavior.fitness))

    def single_step(self):
        self.attempt += 1
        parent = self.attempts[-1][0]
        s = lambda: self.sample(parent)
        samples = {k: v for (k, v) in [s() for _ in range(self.samples)]}
        best = max(samples, key=lambda s: samples[s])
        fitness = samples[best]
        self.attempts.append((best, fitness))
        self.best = self.attempt

    def sample(self, parent: Ctrnn) -> Tuple[Ctrnn, float]:
        ctrnn = parent.clone(self.mutation, self.rng)
        voltages = ctrnn.make_instance()
        behavior = self.new_behavior(ctrnn.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = ctrnn.step(self.dt, voltages)
            behavior.grade(ctrnn.get_output(voltages))
        return ctrnn, behavior.fitness
