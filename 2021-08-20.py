from behavior.oscillator import Oscillator
from concurrent.futures import ProcessPoolExecutor as Pool
from job.learner import Learner
from typing import List
from util.run import Run

import wandb
from util import MetaParameters, get_beers_fitness, get_distance, init_run
from job.nclimber import NClimber
from rl_ctrnn.ctrnn import Ctrnn
import numpy as np


THREAD_COUNT = 10
PROJECT = "undo_perturbed"
GROUP_KEY = "wall_time"
JOB_KEY = "technique"

META = {
    "batch": list(range(1)),
    "technique": ["rl", "hc"],
    "wall_time": [120, 360],
    "mutation_size": [0.25],
    "window": [10],
    "progenitor": [],
}


class Perturbed():
    progenitor: Ctrnn
    perturbed: Ctrnn


class Meta(MetaParameters):
    technique: str
    wall_time: int
    window: float
    mutation_size: float
    progenitor: Ctrnn


def get_frozen(ctrnn: Ctrnn) -> float:
    voltages = ctrnn.make_instance()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=10, window=10)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness


def reinforcement_learner(args: Meta):
    run = init_run(PROJECT, args[GROUP_KEY], args[JOB_KEY], args.__dict__)

    m = Learner(args.progenitor, args.seed)
    m.behavior.dt = 0.01
    m.behavior.duration = args.wall_time
    m.behavior.window = 10

    run.log(
        {
            "Time": 0,
            "fitness": args.progenitor["fitness"],
            "distance": 0,
            "displacement": 0,
            "weights": args.progenitor["weights"],
        }
    )

    time = -1
    while m.behavior.time < m.behavior.duration:
        m.iter()
        if time != (t := np.floor(m.behavior.time)):
            time = t
            run.log(
                {
                    "Time": t,
                    "fitness": get_frozen(m.rlctrnn.ctrnn),
                    "distance": m.rlctrnn.distance,
                    "displacement": m.calculate_displacement(),
                    "weights": m.rlctrnn.ctrnn["weights"],
                }
            )

    data = {
        "distance": m.rlctrnn.distance,
        "displacement": m.calculate_displacement(),
        "initial_fitness": get_beers_fitness(args.progenitor),
        "final_fitness": get_beers_fitness(m.rlctrnn.ctrnn),
        "weights": m.rlctrnn.ctrnn["weights"],
    }
    run.log(data)
    run.finish()


def hill_climber(args: Meta):
    run = init_run(PROJECT, args[GROUP_KEY], args[JOB_KEY], args.__dict__)

    m = NClimber(
        ctrnn=args.progenitor,
        seed=args.seed,
        mutation=args.mutation_size,
        duration=args.window,
    )
    m.setup()

    time = 0
    distance = 0
    run.log(
        {
            "Time": time,
            "fitness": m.attempts[m.best][1],
            "distance": distance,
            "displacement": 0,
            "weights": args.progenitor["weights"],
        }
    )

    a: np.ndarray = m.attempts[0][0].weights
    b: np.ndarray = m.attempts[0][0].weights
    while time < args.wall_time:
        m.single_step()
        b = m.attempts[m.best][0].weights
        distance += get_distance(a, b)
        a = b
        time += m.duration * m.samples
        run.log(
            {
                "Time": time,
                "fitness": m.attempts[m.best][1],
                "distance": distance,
                "displacement": get_distance(args.progenitor.weights, b),
                "weights": m.attempts[-1][0]["weights"],
            }
        )

    fitness = get_beers_fitness(m.attempts[m.best][0])
    data = {
        "initial_fitness": m.attempts[0][1],
        "final_fitness": fitness,
        "weights": m.attempts[m.best][0]["weights"],
    }
    run.log(data)
    run.summary["fitness_ratio"] = data[""]
    run.finish()


def main(args: Meta):
    if args.technique == "rl":
        reinforcement_learner(args)
    if args.technique == "hc":
        hill_climber(args)


if __name__ == "__main__":
    api = wandb.Api()
    runs: List[Run] = api.runs(path="ampersand/perturbed")
    META["progenitor"] = map(lambda r: Ctrnn.from_dict(r.config["ctrnn"]), runs)
    meta = list(map(Meta, Meta.get(META)))
    p = Pool(THREAD_COUNT)
    p.map(main, meta)
