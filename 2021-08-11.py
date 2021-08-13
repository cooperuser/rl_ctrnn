from multiprocessing import Process
from random import randint

from wandb.sdk_py27.wandb_init import init
from util.run import Run
import wandb
from job.nclimber import NClimber
from rl_ctrnn.ctrnn import Ctrnn


COLORS = {
    1: "red",
    2: "orange",
    3: "yellow",
    4: "lime",
    6: "green",
    8: "teal",
    9: "cyan",
    12: "blue",
    18: "purple",
    24: "magenta",
}
FITNESS = 0.5656714076031992
DURATION = 7200
PROGENITOR = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: 5.154455202973727, 1: -10.756384207938911},
    "weights": {
        0: {0: 5.352730101212875, 1: 16.0},
        1: {0: -11.915400080418113, 1: 2.7717190607157542},
    },
}


def fmt(samples: int) -> str:
    zero = "0" if samples < 10 else ""
    return zero + str(samples) + " sample" + ("" if samples == 1 else "s")


def init_run(
    seed: int,
    progenitor: Ctrnn,
    samples: int = 1,
    duration: float = 10,
    mutation: float = 0.05,
) -> Run:
    return wandb.init(
        project="nclimber2",
        group="a",
        job_type=fmt(samples),
        config={
            "seed": seed,
            "progenitor": Ctrnn.to_dict(progenitor),
            "samples": samples,
            "duration": duration,
            "mutation": mutation,
        },
    )


def part_a(
    seed: int = 0, samples: int = 1, duration: float = 10, mutation: float = 0.05
):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run = init_run(seed, progenitor, samples, duration, mutation)
    c = NClimber(
        progenitor, seed=seed, duration=duration, samples=samples, mutation=mutation
    )
    c.setup()
    a = 0
    run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    while a < DURATION:
        c.single_step()
        a += int(c.duration * c.samples)
        ctrnn = c.attempts[c.best][0]
        data = {"Time": a, "fitness": c.attempts[c.best][1]}
        for y in range(ctrnn.size):
            for x in range(ctrnn.size):
                data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
        run.log(data)
    run.finish()
