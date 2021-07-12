from typing import List

import numpy
from evaluator.oscillator import Oscillator
from rl_ctrnn.ctrnn import Ctrnn
from util.run import Run
import wandb

api = wandb.Api()


class Pair:
    def __init__(self, ctrnn: Ctrnn, fitness: float):
        self.ctrnn = ctrnn
        self.fitness = fitness


class Walker(object):
    def __init__(self, id: str, group: str = None):
        self.id = id
        self.group = group
        self.attempts: List[Pair] = []
        self.attempt = 0

        self.run: Run = wandb.init(
            project="rl_ctrnn_test",
            group=self.group,
            job_type="walker",
            config={"progenitor": self.id},
        )

        self.setup()

    def log(self, ctrnn: Ctrnn, fitness: float):
        self.attempts.append(Pair(ctrnn, fitness))
        self.run.log({"ctrnn": Ctrnn.to_dict(ctrnn), "fitness": fitness})

    def setup(self):
        progenitor: Run = api.run("ampersand/rl_ctrnn/" + self.id)
        ctrnn = Ctrnn.from_dict(progenitor.config["ctrnn"], 0.0)
        o = Oscillator(ctrnn)
        o.run()
        self.log(ctrnn, o.fitness / 500)

    def iter(self):
        self.attempt += 1
        parent = self.attempts[-1]
        numpy.random.seed()
        ctrnn = Ctrnn.from_dict(Ctrnn.to_dict(parent.ctrnn), 0.1)
        o = Oscillator(ctrnn)
        o.run()
        self.log(ctrnn, o.fitness / 500)


if __name__ == "__main__":
    c = Walker("7e4amaxv")
    for i in range(2500):
        c.iter()
    c.run.finish()
