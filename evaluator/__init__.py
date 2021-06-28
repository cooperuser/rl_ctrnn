from rl_ctrnn.ctrnn import Ctrnn

class Report(object):
    def __repr__(self):
        return '\n'.join([f"'{k}'\t\t{v}" for k, v in self.__dict__.items()])

class Evaluator(object):
    def __init__(
        self,
        ctrnn: Ctrnn,
        transient_steps: int = 2500,
        evaluation_steps: int = 500,
        time_step: float = 0.1,
    ):
        self.ctrnn = ctrnn
        self._transient_steps = transient_steps
        self._evaluation_steps = evaluation_steps
        self._time_step = time_step
        self._frame = 0

    def _step(self):
        step = self._frame - self._transient_steps;

        if self._frame == 0:
            self.pre_transient(self._time_step, 0)
        elif self._frame == self._transient_steps:
            self.pre_evaluation(self._time_step, 0)

        if self._frame < self._transient_steps:
            self.step_transient(self._time_step, self._frame)
        elif step < self._evaluation_steps:
            self.step_evaluation(self._time_step, step)

        self._frame += 1

    def pre_transient(self, dt: float, step: int):
        pass

    def step_transient(self, dt: float, step: int):
        pass

    def pre_evaluation(self, dt: float, step: int):
        pass

    def step_evaluation(self, dt: float, step: int):
        pass

    def generate_report(self) -> Report:
        return Report()
