from typing_extensions import TypeAlias
from rl_ctrnn.ctrnn import Array, Ctrnn
from typing import List, Tuple

Change: TypeAlias = dict
Evaluation: TypeAlias = List[Tuple[float, Change]]


class Evaluator(object):
    def __init__(self, ctrnn: Ctrnn, durations: Tuple[int, int] = (25, 5)):
        self._time = 0.0
        self._changes: Evaluation = []

        self.dt = 0.05
        self.ctrnn = ctrnn
        self.voltages = self.ctrnn.make_instance()
        self.durations = durations

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

    def grade_transient(self) -> Change:
        return {}

    def grade_evaluation(self) -> Change:
        return {}

    def _track_change(self, change: Change):
        if len(change) == 0:
            return
        self._changes.append((self._time, change))

    def run(self) -> Evaluation:
        """Measure the network's performance for analysis"""
        self.setup_transient()
        t = self.durations[0]
        while self._time < t:
            self.step_transient()
            change = self.grade_transient()
            self._track_change(change)
            self._time += self.dt

        self.setup_evaluation()
        t += self.durations[1]
        while self._time < t:
            self.step_evaluation()
            change = self.grade_evaluation()
            self._track_change(change)
            self._time += self.dt

        return self._changes

    def log(self) -> Evaluation:
        """Measure the network's performance for analysis"""
        self.setup_transient()
        t = self.durations[0]
        while self._time < t:
            self.step_transient()
            change = self.grade_transient()
            self._track_change(change)
            self._time += self.dt

        self.setup_evaluation()
        t += self.durations[1]
        while self._time < t:
            self.step_evaluation()
            change = self.grade_evaluation()
            self._track_change(change)
            self._time += self.dt

        return self._changes

    def get_result(self) -> dict:
        return {}
