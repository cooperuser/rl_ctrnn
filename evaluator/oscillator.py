import numpy as np
import wandb
from rl_ctrnn.ranges import Range
from . import Evaluator, Report

class OscillatorReport(Report):
    def __init__(self):
        self.ranges = [Range(1, 0), Range(1, 0)]
        self.averages = np.zeros(2)
        self.fitness = 0.0
        self.beers_metric = False

class Oscillator(Evaluator):
    def pre_transient(self, dt: float, step: int):
        self.voltages = self.ctrnn.make_instance()
        self.report = OscillatorReport()
        self.fitness = 0

    def step_transient(self, dt: float, step: int):
        self.voltages = self.ctrnn.step(dt, self.voltages)

    def pre_evaluation(self, dt: float, step: int):
        self.report = OscillatorReport()
        self.fitness = 0
        self.last = self.ctrnn.get_output(self.voltages)

    def step_evaluation(self, dt: float, step: int):
        self.voltages = self.ctrnn.step(dt, self.voltages)
        outputs = self.ctrnn.get_output(self.voltages)
        self.fitness += np.sum(abs(outputs + -self.last) / dt)
        self.last = outputs
        for n in range(self.ctrnn.size):
            self.report.ranges[n].set_clamp(outputs[n])
            self.report.averages[n] += outputs[n]

    def generate_report(self) -> OscillatorReport:
        steps = self._evaluation_steps
        self.report.averages /= steps;
        self.report.fitness = self.fitness / (self.ctrnn.size * steps)
        for n in range(self.ctrnn.size):
            neuron = self.report.ranges[n]
            if neuron.max - neuron.min > 0.05:
                self.report.beers_metric = True
                break
        return self.report
