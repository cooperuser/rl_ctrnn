import numpy as np
from rl_ctrnn.ranges import Range
from evaluator import Change, Evaluator


class Oscillator(Evaluator):
    def setup_transient(self):
        self.fitness = 0
        self.variance = 0

    def setup_evaluation(self):
        self.fitness = 0
        self.variance = 0
        self.ranges: dict[int, Range] = {0: Range(1, 0), 1: Range(1, 0)}
        self.last = self.ctrnn.get_output(self.voltages)

    def grade_transient(self, step: int) -> Change:
        outputs = self.ctrnn.get_output(self.voltages)
        return {"a": outputs[0], "b": outputs[1]}

    def grade_evaluation(self, step: int) -> Change:
        outputs = self.ctrnn.get_output(self.voltages)
        self.fitness += np.sum(abs(outputs + -self.last) / self.dt)

        self.variance = 0
        for r, range in self.ranges.items():
            range.set_clamp(outputs[r])
            self.variance += range.max - range.min

        self.last = outputs
        return {
            "fitness": 2 * self.fitness / (self.ctrnn.size * 500),  # TODO: fix this 2 problem
            "variance": self.variance,
            "a": outputs[0],
            "b": outputs[1],
        }
