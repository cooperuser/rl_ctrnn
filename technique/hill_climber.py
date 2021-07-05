from copy import deepcopy
from random import random
from technique import Technique
from rl_ctrnn.ranges import CtrnnRanges

def wiggle(amount: float = 1.0) -> float:
    return (random() * 2 - 1) * amount

class HillClimber(Technique):
    def setup(self):
        self.ctrnn.randomize_weights(self.ranges.weights)
        self.group = "hill_climber_set_bias"
        self.parent = deepcopy(self.ctrnn)
        self.parent_fitness = 0

    def next(self):
        if self.best_fitness > self.parent_fitness:
            self.parent = deepcopy(self.ctrnn)
            self.parent_fitness = self.best_fitness
        self.ctrnn.set_weight(0, 0, self.parent.weights[0][0] + wiggle())
        self.ctrnn.set_weight(0, 1, self.parent.weights[0][1] + wiggle())
        self.ctrnn.set_weight(1, 0, self.parent.weights[1][0] + wiggle())
        self.ctrnn.set_weight(1, 1, self.parent.weights[1][1] + wiggle())


if __name__ == "__main__":
    ranges = CtrnnRanges()
    ranges.set_bias_range(-3, -1)
    ranges.set_time_constant_range(1, 1)
    r = HillClimber(ranges)
    r.start()
    for _ in range(1000):
        r.evaluate()
    r.finish()
