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
    wall_time

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
from rl_ctrnn.ctrnn import Ctrnn
from typing import List

from util import MetaParameters
from util.run import Run
from util.perturbed import Perturbed, get_perturbed_runs

from job.learner import Learner
from job.nclimber import NClimber
import numpy as np
import wandb

THREAD_COUNT = 10
PROJECT = "rl_vs_hc"
DELTA_TIME = 0.01
WINDOW = 10
META = {
    "batch": list(range(1)),
    "perturbed": [],
    "wall_time": [100, 1000],
    "technique": [
        {"name": "rl", "args": {}},
        {"name": "hc", "args": {"mutation_size": 0.25}},
    ],
}


class Meta(MetaParameters):
    technique: dict
    perturbed: Perturbed
    wall_time: int

    def init_run(self) -> Run:
        config = self.perturbed.__dict__.copy()
        config["ctrnn"] = Ctrnn.to_dict(config["ctrnn"])
        config["progenitor"] = Ctrnn.to_dict(config["progenitor"])
        config["technique"] = self.technique["name"]
        config["initial_fitness"] = config["fitness"]
        config["wall_time"] = self.wall_time
        del config["fitness"]
        del config["actual_percent"]
        return wandb.init(
            project=PROJECT,
            group="progenitor_seed",
            job_type="technique",
            config=config,
        )


def run_rl(args: Meta):
    pass


def run_hc(args: Meta):
    pass


def main(args: Meta):
    if args.technique["name"] == "rl":
        run_rl(args)
    elif args.technique["name"] == "hc":
        run_hc(args)
    elif args.technique["name"] == "rw":
        pass


if __name__ == "__main__":
    META["perturbed"] = get_perturbed_runs()
    meta = list(map(Meta, Meta.get(META)))
    meta[0].init_run()
    # p = Pool(THREAD_COUNT)
    # p.map(main, [])
