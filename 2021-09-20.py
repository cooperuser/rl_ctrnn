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
from job.nwalker import NWalker
from rl_ctrnn.ctrnn import Ctrnn
from rl_ctrnn.array_to_dict import array_to_dict

from util import MetaParameters, get_distance, get_beers_fitness, flatdict
from util.perturbed import Perturbed, get_perturbed_runs
from util.run import Run

from job.learner import Learner
from job.nclimber import NClimber

import numpy as np
import wandb

STOP_EARLY_THRESHOLD = -1  # 0.025
THREAD_COUNT = 9
PROJECT = "rl_hc_rw3"
DELTA_TIME = 0.01
WINDOW = 10
META = {
    "batch": list(range(10)),
    "perturbed": [],
    "wall_time": [360],
    "technique": [
        {"name": "rl", "args": {}},
        {"name": "hc", "args": {"mutation_size": 0.25}},
        {"name": "rw", "args": {"mutation_size": 0.25}},
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
            group=self.technique["name"],
            config=flatdict(config),
        )


def run_rl(args: Meta):
    run = args.init_run()

    m = Learner(args.perturbed.ctrnn, args.seed)
    m.behavior.dt = DELTA_TIME
    m.behavior.duration = args.wall_time
    m.behavior.window = WINDOW

    best_fitness = [0.0]

    def log(time: float):
        l = {
            "Time": time,
            "weights": array_to_dict(m.rlctrnn.center),
            "distance_traveled": m.rlctrnn.distance,
            "displacement": {
                "start": m.calculate_displacement(),
                "progenitor": get_distance(
                    m.rlctrnn.center, args.perturbed.progenitor.weights
                ),
            },
        }
        if time % 10 == 0:
            l["fitness"] = get_beers_fitness(m.rlctrnn.ctrnn, 10, 10)
            best_fitness[0] = max(l["fitness"], best_fitness[0])
            l["best_fitness"] = best_fitness[0]
        run.log(flatdict(l))

    log(time := 0)
    while time < m.behavior.duration:
        m.iter()
        if time != (t := np.floor(m.behavior.time * 10) / 10):
            time = t
            log(t)
            if m.rlctrnn.flux < STOP_EARLY_THRESHOLD:
                break

    fitness = get_beers_fitness(m.rlctrnn.ctrnn)
    run.summary["beers_fitness"] = fitness
    run.summary["fitness_ratio"] = fitness / args.perturbed.progenitor_fitness
    run.finish()


def run_hc(args: Meta, Type):
    run = args.init_run()

    m = Type(
        ctrnn=args.perturbed.ctrnn,
        seed=args.seed,
        mutation=args.technique["args"]["mutation_size"],
        duration=WINDOW,
        dt=DELTA_TIME,
    )
    m.setup()

    distance = 0

    def log(time: float):
        ctrnn = m.attempts[m.best][0]
        l = {
            "Time": time,
            "fitness": m.attempts[-1][1],
            "best_fitness": m.attempts[m.best][1],
            "weights": array_to_dict(ctrnn.weights),
            "distance_traveled": distance,
            "displacement": {
                "start": get_distance(ctrnn.weights, args.perturbed.ctrnn.weights),
                "progenitor": get_distance(
                    ctrnn.weights, args.perturbed.progenitor.weights
                ),
            },
        }
        run.log(flatdict(l))

    log(time := 0)
    while time < args.wall_time:
        a = m.attempts[m.best][0].weights
        m.single_step()
        b = m.attempts[m.best][0].weights
        distance += get_distance(a, b)
        time += m.duration * m.samples
        log(time)

    fitness = get_beers_fitness(m.attempts[m.best][0], 10, 10)
    run.summary["beers_fitness"] = fitness
    run.summary["fitness_ratio"] = fitness / args.perturbed.progenitor_fitness
    run.finish()


def main(args: Meta):
    if args.technique["name"] == "rl":
        run_rl(args)
    elif args.technique["name"] == "hc":
        run_hc(args, NClimber)
    elif args.technique["name"] == "rw":
        run_hc(args, NWalker)


if __name__ == "__main__":
    META["perturbed"] = list(
        filter(
            lambda x: x.progenitor_mutation_size == 0.25 and x.target_percent == 0.1,
            get_perturbed_runs(),
        )
    )
    meta = list(map(Meta, Meta.get(META)))
    p = Pool(THREAD_COUNT)
    p.map(main, meta)
