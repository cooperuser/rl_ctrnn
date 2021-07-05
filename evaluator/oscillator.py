from typing import Dict
import numpy as np
import wandb
from rl_ctrnn.ranges import Range
from . import Evaluator, Report

class OscillatorReport(Report):
    def __init__(self):
        self.ranges = [Range(1, 0), Range(1, 0)]
        self.averages = np.zeros(2)
        self.fitness = 0.0
        self.beers_metric = 0

class Oscillator(Evaluator):
    def reset(self):
        self.voltages = self.ctrnn.make_instance()
        self.report = OscillatorReport()
        self.fitness = 0

    def pre_transient(self, dt: float, step: int):
        self.voltages = self.ctrnn.make_instance()
        self.report = OscillatorReport()
        self.fitness = 0

    def step_transient(self, dt: float, step: int):
        self.voltages = self.ctrnn.step(dt, self.voltages)

    def pre_evaluation(self, dt: float, step: int):
        self.report = OscillatorReport()
        self.fitness = 0
        self.beers_metric = 0
        self.last = self.ctrnn.get_output(self.voltages)

    def step_evaluation(self, dt: float, step: int):
        self.voltages = self.ctrnn.step(dt, self.voltages)
        outputs = self.ctrnn.get_output(self.voltages)
        self.fitness += np.sum(abs(outputs + -self.last) / dt)
        beers = sum(map(lambda x: x.max - x.min,self.report.ranges))
        if beers > self.beers_metric:
            self.beers_metric = beers
        self.last = outputs
        for n in range(self.ctrnn.size):
            self.report.averages[n] += outputs[n]
            self.report.ranges[n].set_clamp(outputs[n])
            neuron = self.report.ranges[n]
            if not self.report.beers_metric and neuron.max - neuron.min > 0.05:
                self.report.beers_metric = step

    def generate_report(self) -> OscillatorReport:
        steps = self._evaluation_steps
        self.report.averages /= steps;
        self.report.fitness = self.fitness / (self.ctrnn.size * steps)
        return self.report

    def get_log(self) -> Dict:
        return {
            "fitness": self.report.fitness,
            "beers_metric": self.report.beers_metric,
            "weight_0_0": self.ctrnn.weights[0][0],
            "weight_0_1": self.ctrnn.weights[0][1],
            "weight_1_0": self.ctrnn.weights[1][0],
            "weight_1_1": self.ctrnn.weights[1][1],
        }
