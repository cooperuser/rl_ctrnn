from concurrent.futures import ProcessPoolExecutor as Pool
from util import MetaParameters, get_beers_fitness, get_distance, init_run
from job.nclimber import NClimber
from rl_ctrnn.ctrnn import Ctrnn
import numpy as np


THREAD_COUNT = 10
PROJECT = "mutation_vs_wall"
GROUP_KEY = "wall_time"
JOB_KEY = "mutation_size"
META = {
    "batch": list(range(19)),
    "wall_time": [120, 360, 1200],
    "mutation_size": [0.05, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "window": [10],
    "progenitor": map(
        Ctrnn.from_dict,
        [
            {
                "time_constants": {0: 1.0, 1: 1.0},
                "biases": {0: 5.154455202973727, 1: -10.756384207938911},
                "weights": {
                    0: {0: 5.352730101212875, 1: 16.0},
                    1: {0: -11.915400080418113, 1: 2.7717190607157542},
                },
            }
        ],
    ),
}


class Meta(MetaParameters):
    wall_time: int
    window: float
    mutation_size: float
    progenitor: Ctrnn


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

    data = {
        "initial_fitness": get_beers_fitness(args.progenitor),
        "final_fitness": get_beers_fitness(m.attempts[m.best][0]),
        "weights": m.attempts[m.best][0]["weights"],
    }
    run.log(data)
    run.finish()


if __name__ == "__main__":
    meta = list(map(Meta, Meta.get(META)))
    p = Pool(THREAD_COUNT)
    p.map(hill_climber, meta)
