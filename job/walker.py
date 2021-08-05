from typing_extensions import TypeAlias
from behavior.oscillator import Oscillator
from typing import List, Tuple
from rl_ctrnn.ctrnn import Ctrnn


Pair: TypeAlias = Tuple[Ctrnn, float]
# class Pair:
#     def __init__(self, ctrnn: Ctrnn, fitness: float):
#         self.ctrnn = ctrnn
#         self.fitness = fitness


class Walker(object):
    def __init__(
        self, ctrnn: Ctrnn, seed: int = 0, mutation: float = 0.05, dt: float = 0.01
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.mutation = mutation
        self.attempts: List[Pair] = []
        self.attempt = 0
        self.best = 0
        self.dt = dt

    def setup(self):
        behavior = Oscillator(dt=self.dt, size=self.progenitor.size)
        voltages = self.progenitor.make_instance()
        behavior.setup(self.progenitor.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = self.progenitor.step(self.dt, voltages)
            behavior.grade(self.progenitor.get_output(voltages))
        self.attempts.append((self.progenitor, behavior.fitness))

    def iter(self):
        self.attempt += 1
        parent = self.attempts[-1]
        ctrnn = Ctrnn.from_dict(Ctrnn.to_dict(parent[0]), self.mutation)
        behavior = Oscillator(dt=self.dt, size=ctrnn.size)
        voltages = ctrnn.make_instance()
        behavior.setup(ctrnn.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = ctrnn.step(self.dt, voltages)
            behavior.grade(ctrnn.get_output(voltages))
        self.attempts.append((ctrnn, behavior.fitness))
        if behavior.fitness >= self.attempts[self.best][1]:
            self.best = self.attempt
