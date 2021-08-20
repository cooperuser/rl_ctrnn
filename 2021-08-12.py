from multiprocessing.context import Process
from random import randint
from job.nclimber import NClimber
import wandb
from util.run import Run
from rl_ctrnn.ctrnn import Ctrnn

THREAD_COUNT = 10
FITNESS = 0.5656714076031992
DURATION = 360
PROGENITOR = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: 5.154455202973727, 1: -10.756384207938911},
    "weights": {
        0: {0: 5.352730101212875, 1: 16.0},
        1: {0: -11.915400080418113, 1: 2.7717190607157542},
    },
}


def init_run(
    group: str,
    seed: int,
    progenitor: dict = PROGENITOR,
    samples: int = 1,
    duration: float = 10,
    mutation: float = 0.05,
) -> Run:
    return wandb.init(
        project="nclimber",
        group=group,
        config={
            "seed": seed,
            "progenitor": progenitor,
            "samples": samples,
            "duration": duration,
            "mutation": mutation,
        },
    )


def part_a(mutation: float = 0.05, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run = init_run("a3", seed, mutation=mutation)
    c = NClimber(progenitor, seed=seed, duration=10, samples=1, mutation=mutation)
    c.setup()
    a = 0
    run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    while a < DURATION:
        c.single_step()
        a += int(c.duration * c.samples)
        data = {"Time": a, "fitness": c.attempts[c.best][1]}
        ctrnn = c.attempts[c.best][0]
        for y in range(ctrnn.size):
            for x in range(ctrnn.size):
                data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
        run.log(data)
    run.finish()


if __name__ == "__main__":
    n = 9
    for t in range(n):
        for mut in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            print(t / n, mut)
            threads = []
            for _ in range(10):
                seed = randint(1, 100000)
                threads.append(Process(target=part_a, args=(mut, seed,)))
            for _, p in enumerate(threads):
                p.start()
            for _, p in enumerate(threads):
                p.join()
