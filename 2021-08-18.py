from job.nclimber import NClimber
from math import ceil
from multiprocessing.context import Process
from behavior.oscillator import Oscillator
from multiprocessing import Pool
from util.run import Run
from numpy import floor
import wandb
from job.learner import Learner
from rl_ctrnn.ctrnn import Ctrnn
import itertools
import random
import numpy as np


THREAD_COUNT = 10
WALL_TIMES = [120, 360, 1200]
SEEDS = [random.randint(100000, 999999) for _ in range(50)]
PROGENITORS = [
    Ctrnn.from_dict(
        {
            "time_constants": {0: 1.0, 1: 1.0},
            "biases": {0: 5.154455202973727, 1: -10.756384207938911},
            "weights": {
                0: {0: 5.352730101212875, 1: 16.0},
                1: {0: -11.915400080418113, 1: 2.7717190607157542},
            },
        }
    )
]


def init_run(job, wall_time, progenitor, seed) -> Run:
    return wandb.init(
        project="2021-08-19",
        group=str(wall_time),
        job_type=job,
        config={"wall_time": wall_time, "progenitor": progenitor, "seed": seed},
    )


def get_frozen(ctrnn: Ctrnn) -> float:
    voltages = ctrnn.make_instance()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=10, window=10)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness


def calculate_displacement(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(a + -b, 2)))


def hill_climber(wall_time, ctrnn, seed):
    run = init_run("hill_climber", wall_time, ctrnn, seed)

    m = NClimber(ctrnn, seed, mutation=0.15)
    m.setup()

    time = 0
    distance = 0
    data = {
        "Time": time,
        "fitness": m.attempts[m.best][1],
        "distance": 0,
        "displacement": 0,
    }
    for y in range(ctrnn.size):
        for x in range(ctrnn.size):
            data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
    run.log(data)
    a: np.ndarray = m.attempts[0][0].weights
    b: np.ndarray = m.attempts[0][0].weights
    while time < wall_time:
        m.single_step()
        b = m.attempts[m.best][0].weights
        distance += calculate_displacement(a, b)
        a = b
        time += int(m.duration * m.samples)
        data = {
            "Time": time,
            "fitness": m.attempts[m.best][1],
            "distance": distance,
            "displacement": calculate_displacement(ctrnn.weights, b),
        }
        c = m.attempts[m.best][0]
        for y in range(c.size):
            for x in range(c.size):
                data[f"weight.{x}.{y}"] = c.weights[x, y]
        run.log(data)

    run.finish()


def reinforcement_learner(wall_time, ctrnn, seed):
    run = init_run("rl_learner", wall_time, ctrnn, seed)

    m = Learner(ctrnn, seed)
    m.behavior.dt = 0.01
    m.behavior.duration = wall_time
    m.behavior.window = 10

    time = -1
    while m.behavior.time < m.behavior.duration:
        m.iter()
        if time != (t := floor(m.behavior.time)):  # possible rename `t` to `time`
            if t % int(wall_time / 120):
                continue
            time = t
            data = {"Time": t}
            data["fitness"] = get_frozen(m.rlctrnn.ctrnn)  # m.behavior.fitness
            data["distance"] = m.rlctrnn.distance
            data["displacement"] = m.calculate_displacement()
            for y in range(m.rlctrnn.ctrnn.size):
                for x in range(m.rlctrnn.ctrnn.size):
                    data[f"weight.{x}.{y}"] = m.rlctrnn.center[x, y]
            run.log(data)

    # run.summary["fitness"] = get_frozen(m.rlctrnn.ctrnn)
    run.finish()


if __name__ == "__main__":

    def c(args):
        print(args)
        args[0](*args[1:])

    # methods = [random_walker, hill_climber, reinforcement_learner]
    args = itertools.product(
        [reinforcement_learner, hill_climber], WALL_TIMES, PROGENITORS, SEEDS
    )
    # p = Pool(2)
    # p.map(c, list(args))
    args = list(args)
    num_threads = ceil(len(args) / THREAD_COUNT)
    groups = [args[i::num_threads] for i in range(num_threads)]
    for group in groups:
        threads = []
        for g in group:
            threads.append(Process(target=c, args=(g,)))
        for _, p in enumerate(threads):
            p.start()
        for _, p in enumerate(threads):
            p.join()
