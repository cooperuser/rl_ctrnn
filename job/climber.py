import wandb
from util.run import Run
import numpy as np
from evaluator.oscillator import Oscillator, OscillatorResult
from rl_ctrnn.ctrnn import Ctrnn

from job import *


class Pair:
    def __init__(self, ctrnn: Ctrnn, result: OscillatorResult):
        self.ctrnn = ctrnn
        self.fitness = result.fitness
        self.variance = result.variance


class Climber(object):
    def __init__(
        self,
        project: str,
        group: str,
        ctrnn: Ctrnn,
        seed: int = 0,
        mutation_size: float = 0.1,
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.mutation = mutation_size
        self.attempts: List[Pair] = []
        self.attempt = 0
        self.best = 0

        self.run: Run = wandb.init(
            project=project,
            group=group,
            job_type="climber",
            config={
                "seed": self.seed,
                "mutation_size": mutation_size,
                "progenitor": Ctrnn.to_dict(self.progenitor),
            },
        )

        np.random.seed(self.seed)
        self.setup()

    def log(self, ctrnn: Ctrnn, result: OscillatorResult):
        self.attempts.append(Pair(ctrnn, result))
        if result.fitness >= self.attempts[self.best].fitness:
            self.best = self.attempt
        self.run.log(
            {
                "ctrnn": Ctrnn.to_dict(ctrnn),
                "fitness": result.fitness,
                "variance": result.variance,
            }
        )

    def setup(self):
        o = Oscillator(self.progenitor)
        o.run()
        self.log(self.progenitor, o.get_result())

    def iter(self):
        self.attempt += 1
        parent = self.attempts[self.best]
        ctrnn = Ctrnn.from_dict(Ctrnn.to_dict(parent.ctrnn), self.mutation)
        o = Oscillator(ctrnn)
        o.run()
        self.log(ctrnn, o.get_result())
