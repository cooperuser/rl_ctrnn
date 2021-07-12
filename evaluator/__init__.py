from typing_extensions import TypeAlias
from rl_ctrnn.ctrnn import Array, Ctrnn
from rl_ctrnn.ranges import CtrnnRanges
from typing import List, Tuple

Change: TypeAlias = dict
Evaluation: TypeAlias = List[Tuple[int, Change]]


class Evaluator(object):
    def __init__(self, ctrnn: Ctrnn):
        self._step = 0
        self._changes: Evaluation = []

        self.dt = 0.01
        self.ctrnn = ctrnn
        self.voltages = self.ctrnn.make_instance()

    def setup(self):
        pass

    def setup_transient(self):
        self.setup()

    def setup_evaluation(self):
        self.setup()

    def step(self) -> Array:
        self.voltages = self.ctrnn.step(self.dt, self.voltages)
        return self.ctrnn.get_output(self.voltages)

    def step_transient(self) -> Array:
        return self.step()

    def step_evaluation(self) -> Array:
        return self.step()

    def grade_transient(self, step: int) -> Change:
        return {}

    def grade_evaluation(self, step: int) -> Change:
        return {}

    def _track_change(self, change: Change, phase: int = 0):
        if len(change) == 0:
            return
        # change["phase"] = phase
        # if phase == 0 and self._step % 10 != 0:
        #     return
        # if phase == 1 and self._step % 5 != 0:
        #     return
        self._changes.append((self._step, change))

    def run(self) -> Evaluation:
        """Measure the network's performance for analysis"""
        self.setup_transient()
        for self._step in range(0, 2500):
            self.step_transient()
            change = self.grade_transient(self._step)
            self._track_change(change, 0)

        self.setup_evaluation()
        for self._step in range(2500, 3000):
            self.step_evaluation()
            change = self.grade_evaluation(self._step)
            self._track_change(change, 1)

        return self._changes
