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
# from typing import List

from util import MetaParameters, get_distance, get_beers_fitness
from util.perturbed import Perturbed, get_perturbed_runs
from util.run import Run

from job.learner import Learner
# from job.nclimber import NClimber
import numpy as np
import wandb

STOP_EARLY_THRESHOLD = -1  # 0.025
THREAD_COUNT = 10
PROJECT = "rl_vs_hc"
DELTA_TIME = 0.01
WINDOW = 10
META = {
    "batch": list(range(10)),
    "perturbed": [],
    "wall_time": [1000],
    "technique": [
        {"name": "rl", "args": {}},
        # {"name": "hc", "args": {"mutation_size": 0.25}},
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
            group=str(self.perturbed.progenitor_seed),
            job_type=self.technique["name"],
            config=config,
        )


def run_rl(args: Meta):
    run = args.init_run()

    m = Learner(args.perturbed.ctrnn, args.seed)
    m.behavior.dt = DELTA_TIME
    m.behavior.duration = args.wall_time
    m.behavior.window = WINDOW

    def log(time: float):
        e, c = (enumerate, m.rlctrnn.center)
        weights = {a: {b: w for b, w in e(ws)} for a, ws in e(c)}
        run.log(
            {
                "Time": time,
                # "flux": m.rlctrnn.flux,
                "weights": weights,
                "distance_traveled": m.rlctrnn.distance,
                "displacement": {
                    "start": m.calculate_displacement(),
                    "progenitor": get_distance(
                        m.rlctrnn.center, args.perturbed.progenitor.weights
                    ),
                },
            }
        )

    log(0)
    time = -1
    while time < m.behavior.duration:
        m.iter()
        if time != (t := np.floor(m.behavior.time * 10) / 10):
            time = t
            log(t)
            if m.rlctrnn.flux < STOP_EARLY_THRESHOLD:
                break

    fitness = get_beers_fitness(m.rlctrnn.ctrnn)
    run.summary["final_fitness"] = fitness
    run.summary["fitness_ratio"] = fitness / args.perturbed.progenitor_fitness
    run.finish()


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
    print(len(meta))
    # main(meta[0])
    p = Pool(THREAD_COUNT)
    p.map(main, meta)
