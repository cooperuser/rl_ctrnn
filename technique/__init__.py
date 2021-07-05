from evaluator import Report
import wandb
from copy import deepcopy
from evaluator.oscillator import Oscillator
from rl_ctrnn.ranges import CtrnnRanges
from rl_ctrnn.ctrnn import Ctrnn

class Technique(object):
    def __init__(self, ranges: CtrnnRanges, Evaluator = Oscillator):
        self.ranges = ranges
        self.Evaluator = Evaluator
        self.ctrnn = Ctrnn(2)
        self.ctrnn.randomize_biases(ranges.biases)
        self.ctrnn.randomize_time_constants(ranges.time_constants)
        self.time_step = 0.01
        self.group = ""
        self.report = None
        self.best_fitness = 0
        self.best = {}
        self.setup()

    def setup(self):
        pass

    def next(self):
        pass

    def get_log(self):
        if self.report == None:
            return {"fitness": 0}
        log = {
            "fitness": self.report.fitness,
            "beers_metric": self.report.beers_metric,
            "weight_0_0": self.ctrnn.weights[0][0],
            "weight_0_1": self.ctrnn.weights[0][1],
            "weight_1_0": self.ctrnn.weights[1][0],
            "weight_1_1": self.ctrnn.weights[1][1],
        }
        return log

    def evaluate(self):
        self.next()
        eval = self.Evaluator(self.ctrnn, time_step=0.01)
        for _ in range(3000):
            eval._step()
        self.report = eval.generate_report()
        log = eval.get_log()
        if log["fitness"] > self.best_fitness:
            self.best_fitness = log["fitness"]
            self.best = deepcopy(log)
        log["best"] = self.best
        wandb.log(log)

    def start(self):
        wandb.init(entity="ampersand", project="hill_climber", config={
            "group": self.group,
            "time_step": self.time_step,
            "bias_a": self.ctrnn.biases[0],
            "bias_b": self.ctrnn.biases[1],
            "time_constant_a": self.ctrnn.time_constants[0],
            "time_constant_b": self.ctrnn.time_constants[1]
        })

    def finish(self):
        wandb.finish()
