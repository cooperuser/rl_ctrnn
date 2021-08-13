from job.nwalker import NWalker
from multiprocessing import Process
from random import randint
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


def part_a(samples: int = 1, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run: Run = wandb.init(
        project="nclimber",
        group="a",
        job_type=fmt(samples),
        config={"samples": samples, "seed": seed},
    )
    c = NClimber(progenitor, seed=seed, duration=10, samples=samples)
    c.setup()
    a = 0
    run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    while a < DURATION:
        c.single_step()
        a += int(c.duration * c.samples)
        run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    run.finish()


def part_b(samples: int = 1, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run: Run = wandb.init(
        project="nclimber",
        group="b",
        job_type=fmt(samples),
        config={"samples": samples, "seed": seed},
    )
    c = NClimber(
        progenitor, seed=seed, duration=10, samples=samples, mutation=0.05 * samples
    )
    c.setup()
    a = 0
    run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    while a < DURATION:
        c.single_step()
        a += int(c.duration * c.samples)
        run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    run.finish()


def part_c(mutation: float = 0.05, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run: Run = wandb.init(
        project="nclimber",
        group="c",
        job_type=fmt(8),
        config={"samples": 8, "seed": seed, "mutation": mutation},
    )
    c = NClimber(
        progenitor, seed=seed, duration=10, samples=8, mutation=mutation
    )
    c.setup()
    a = 0
    run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    while a < DURATION:
        c.single_step()
        a += int(c.duration * c.samples)
        run.log({"Time": a, "fitness": c.attempts[c.best][1]})
    run.finish()


def part_f(mutation: float = 0.05, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run: Run = wandb.init(
        project="nclimber",
        group="f2",
        job_type=fmt(8),
        config={"samples": 1, "seed": seed, "mutation": mutation},
    )
    c = NClimber(
        progenitor, seed=seed, duration=10, samples=1, mutation=mutation
    )
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


def part_g(mutation: float = 0.05, seed: int = 0):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    run: Run = wandb.init(
        project="nclimber",
        group="g",
        job_type=fmt(8),
        config={"samples": 1, "seed": seed, "mutation": mutation},
    )
    c = NWalker(
        progenitor, seed=seed, duration=10, samples=1, mutation=mutation
    )
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
    loops = 10.0
    for n in range(int(loops)):
        print(n / loops)
        threads = []
        seed = randint(11, 50000)
        for i in [0.05, 0.1, 0.5, 1, 2, 3, 4, 5]:
            threads.append(Process(target=part_f, args=(i, seed)))
        for _, p in enumerate(threads):
            p.start()
        for _, p in enumerate(threads):
            p.join()
