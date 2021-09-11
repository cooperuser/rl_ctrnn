from util import get_beers_fitness, get_distance, init_run
from job.nclimber import NClimber
import numpy as np
# import wandb

THREAD_COUNT = 10
PROJECT = "undo_perturbed"
GROUP_KEY = "wall_time"
JOB_KEY = "technique"
# META = {
#     "batch": list(range(1)),
#     "progenitor": [],
# }



def hill_climber(args: Meta):
    run = init_run(PROJECT, args[GROUP_KEY], args[JOB_KEY], args.__dict__)

    m = NClimber(
        ctrnn=args.progenitor,
        seed=args.seed,
        mutation=args.mutation_size,
        duration=args.window,
    )
    m.setup()

    # distance = 0
    run.log(
        {
            "fitness": m.attempts[m.best][1],
            # "distance": distance,
            # "displacement": 0,
            "weights": args.progenitor["weights"],
        }
    )

    # a: np.ndarray = m.attempts[0][0].weights
    # b: np.ndarray = m.attempts[0][0].weights
    for _ in range(100):
        m.single_step()
        # b = m.attempts[m.best][0].weights
        # distance += get_distance(a, b)
        # a = b
        run.log(
            {
                "fitness": m.attempts[m.best][1],
                # "distance": distance,
                # "displacement": get_distance(args.progenitor.weights, b),
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


if __name__ == "__main__":
    # api = wandb.Api()
    # runs: List[Run] = api.runs(path="ampersand/perturbed")
    # META["progenitor"] = map(lambda r: Ctrnn.from_dict(r.config["ctrnn"]), runs)
    meta = list(map(Meta, Meta.get(META)))
    p = Pool(THREAD_COUNT)
    p.map(main, meta)
