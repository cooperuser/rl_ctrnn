from multiprocessing import Process
from random import randint
from rl_ctrnn.ctrnn import Ctrnn
from job.climber import Climber
from job.walker import Walker
from job.learner import Learner
from util.run import Run
import numpy as np
import wandb


PROJECT = "all-three"
GROUP = "one"
TIMESTEP = 0.01
MUTATION_SIZE = 0.05
ITERATIONS = 120
THREAD_COUNT = 10
PROGENITOR = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: 5.154455202973727, 1: -10.756384207938911},
    "weights": {
        0: {0: 4.6936547442550935, 1: 12.453608462321759},
        1: {0: -12.842427683843608, 1: 4.044041822249799},
    },
}
OPTIMAL = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: 5.154455202973727, 1: -10.756384207938911},
    "weights": {
        0: {0: 4.901027283936059, 1: 16.0},
        1: {0: -16, 1: 4.71474351414656},
    },
}

WEIGHTS_PROGENITOR: np.ndarray = Ctrnn.from_dict(PROGENITOR).weights
WEIGHTS_OPTIMAL: np.ndarray = Ctrnn.from_dict(OPTIMAL).weights


def init_run(job: str, seed: int, mut: float = MUTATION_SIZE) -> Run:
    return wandb.init(
        project=PROJECT,
        job_type=GROUP,
        group=job,
        config={
            "optimal": OPTIMAL,
            "progenitor": PROGENITOR,
            "seed": seed,
            "mutation_size": mut,
        },
    )


def calculate_displacement(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(a + -b, 2)))


def walker(seed: int):
    run = init_run("walker", seed)
    progenitor = Ctrnn.from_dict(PROGENITOR)
    distance = 0.0
    m = Walker(progenitor, seed, MUTATION_SIZE, TIMESTEP)
    m.setup()
    for i in range(ITERATIONS):
        m.iter()
        a: np.ndarray = m.attempts[-2][0].weights
        b: np.ndarray = m.attempts[-1][0].weights
        distance += calculate_displacement(a, b)
        ctrnn, fitness = m.attempts[-1]
        data = {"Time": float(i)}
        data["fitness"] = m.attempts[m.best][1]
        data["distance"] = distance
        d_a = calculate_displacement(ctrnn.weights, WEIGHTS_PROGENITOR)
        d_b = calculate_displacement(ctrnn.weights, WEIGHTS_OPTIMAL)
        data["displacement"] = d_a
        data["remaining"] = d_b

        data["temp_fitness"] = fitness

        for y in range(ctrnn.size):
            for x in range(ctrnn.size):
                data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
        run.log(data)
    run.finish()


def climber(seed: int):
    run = init_run("climber", seed)
    progenitor = Ctrnn.from_dict(PROGENITOR)
    distance = 0.0
    m = Climber(progenitor, seed, MUTATION_SIZE, TIMESTEP)
    m.setup()
    for i in range(ITERATIONS):
        m.iter()
        a: np.ndarray = m.attempts[-2][0].weights
        b: np.ndarray = m.attempts[-1][0].weights
        distance += calculate_displacement(a, b)
        ctrnn, fitness = m.attempts[-1]
        data = {"Time": float(i)}
        data["fitness"] = m.attempts[m.best][1]
        data["distance"] = distance
        d_a = calculate_displacement(ctrnn.weights, WEIGHTS_PROGENITOR)
        d_b = calculate_displacement(ctrnn.weights, WEIGHTS_OPTIMAL)
        data["displacement"] = d_a
        data["remaining"] = d_b

        data["temp_fitness"] = fitness

        for y in range(ctrnn.size):
            for x in range(ctrnn.size):
                data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
        run.log(data)
    run.finish()


def learner(seed: int):
    run = init_run("learner", seed, 0.0)
    progenitor = Ctrnn.from_dict(PROGENITOR)
    m = Learner("", "", progenitor, seed)
    while m.behavior.time < m.behavior.duration:
        m.iter()
        data = {"Time": m.behavior.time}
        data["fitness"] = m.behavior.fitness
        data["distance"] = m.rlctrnn.distance
        data["displacement"] = m.calculate_displacement()
        d_b = calculate_displacement(m.rlctrnn.center, WEIGHTS_OPTIMAL)
        data["remaining"] = d_b

        data["performance"] = m.performance
        data["reward"] = m.reward
        data["flux"] = m.rlctrnn.flux

        for y in range(m.rlctrnn.ctrnn.size):
            for x in range(m.rlctrnn.ctrnn.size):
                data[f"weight.{x}.{y}"] = m.rlctrnn.center[x, y]
        run.log(data)
    run.finish()


def main(job):
    threads = []
    for _ in range(THREAD_COUNT):
        seed = randint(1, 100000)
        threads.append(Process(target=job, args=(seed,)))
    for _, p in enumerate(threads):
        p.start()
    for _, p in enumerate(threads):
        p.join()


if __name__ == "__main__":
    for _ in range(1):
        main(walker)
        main(climber)
        main(learner)
