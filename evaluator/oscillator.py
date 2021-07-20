import numpy as np
from rl_ctrnn.ranges import Range
from evaluator import Change, Evaluator


class OscillatorResult(object):
    def __init__(self, fitness: float, variance: float):
        self.fitness = fitness
        self.variance = variance

    def to_dict(self) -> dict:
        return {"fitness": self.fitness, "variance": self.variance}


class Oscillator(Evaluator):
    def setup_transient(self):
        self.fitness = 0
        self.variance = 0

    def setup_evaluation(self):
        self.fitness = 0
        self.variance = 0
        self.ranges: dict[int, Range] = {0: Range(1, 0), 1: Range(1, 0)}
        self.last = self.ctrnn.get_output(self.voltages)

    def grade_transient(self) -> Change:
        outputs = self.ctrnn.get_output(self.voltages)
        return {"a": outputs[0], "b": outputs[1]}

    def grade_evaluation(self) -> Change:
        outputs = self.ctrnn.get_output(self.voltages)
        self.fitness += np.sum(abs(outputs + -self.last))

        self.variance = 0
        for r, range in self.ranges.items():
            range.set_clamp(outputs[r])
            self.variance += range.max - range.min

        self.last = outputs
        d = self.ctrnn.size * self.durations[1] * self.dt
        return {
            "fitness": self.fitness / d,
            "variance": self.variance,
            "a": outputs[0],
            "b": outputs[1],
        }

    def get_result(self) -> OscillatorResult:
        return OscillatorResult(
            self.fitness / (self.ctrnn.size * self.durations[1] * self.dt),
            self.variance,
        )
