"""
Runs rl, hc, rw instances from each of the perturbed networks

Group: progenitor_seed
JobType: technique

Config:
    ctrnn
    progenitor
    progenitor_seed
    progenitor_mutation_size
    progenitor_fitness
    initial_fitness
    target_percent
    technique

History metrics:
    weights
    distance_traveled
    displacement:
        start
        end
        progenitor

Summary metrics:
    final_fitness
    fitness_ratio
"""

from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List

from util import MetaParameters
from util.run import Run
from util.perturbed import Perturbed, get_perturbed_runs

from job.learner import Learner
from job.nclimber import NClimber
import numpy as np
import wandb

THREAD_COUNT = 10
PROJECT = ""
META = {
    "batch": list(range(1)),
    "perturbed": [],
    "dt": 0.01,
    "wall_time": 1000,
    "window": 10,
    "technique": [
        {"name": "rl", "args": {}},
        {"name": "hc", "args": {"mutation_size": 0.25}},
    ],
}


class Meta(MetaParameters):
    technique: dict
    perturbed: Perturbed
    dt: float
    wall_time: float
    window: float


def main(args: Meta):
    if args.technique["name"] == "rl":
        pass
    elif args.technique["name"] == "hc":
        pass
    elif args.technique["name"] == "rw":
        pass


if __name__ == "__main__":
    META["perturbed"] = get_perturbed_runs()
    meta = list(map(Meta, Meta.get(META)))
    print(meta[0])
    # p = Pool(THREAD_COUNT)
    # p.map(main, [])
