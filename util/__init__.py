import itertools
import random
from typing import Any, List, Mapping

import wandb
from rl_ctrnn.ctrnn import Ctrnn
from behavior.oscillator import Oscillator
from .run import Run
from .seed import Seed
import numpy as np
from flatdict import FlatDict


def get_beers_fitness(ctrnn: Ctrnn, duration: float = 300, window: float = 50) -> float:
    voltages = ctrnn.make_instance()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=duration, window=window)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness


def get_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(a + -b, 2)))


def init_run(project: str, group: Any, job: Any, args: Mapping) -> Run:
    return wandb.init(
        project=project,
        group=str(group),
        job_type=str(job),
        config=args,
    )


class MetaParameters(Mapping):
    def __init__(self, args: dict):
        self.seed: int = random.randint(100000, 999999)
        for k, v in args.items():
            self[k] = v

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def get(meta: dict) -> List[dict]:
        keys = list(meta.keys())
        values = list(itertools.product(*meta.values()))
        return [{keys[i]: v[i] for i in range(len(keys))} for v in values]

def flatdict(d: dict) -> dict:
    return dict(FlatDict(d, delimiter=':'))
