from collections import deque
from multiprocessing.context import Process
from random import randint
from time import perf_counter
from rl_ctrnn.rl_ctrnn import RLCtrnn
from util.run import Run
from behavior.oscillator import Oscillator
from rl_ctrnn.ctrnn import Array, Ctrnn
import numpy as np
import wandb

# class Pair:
#     def __init__(self, ctrnn: Ctrnn, result: OscillatorResult):
#         self.ctrnn = ctrnn
#         self.fitness = result.fitness
#         self.variance = result.variance


class Learner(object):
    def __init__(
        self,
        project: str,
        group: str,
        ctrnn: Ctrnn,
        seed: int = 0,
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.behavior = Oscillator(self.progenitor.size)
        prog = Ctrnn.to_dict(ctrnn)
        self.rlctrnn = RLCtrnn(Ctrnn.from_dict(prog), self.seed)
        self.voltages = self.rlctrnn.ctrnn.make_instance()
        self.time = 0

        self.count = int(self.behavior.durations[1] / self.behavior.dt)
        self.performances = deque([0.0 for _ in range(self.count)])
        self.rewards = deque([0.0 for _ in range(self.count)])
        self.fitness = 0
        self.total_performance = 0
        self.total_reward = 0
        self.behavior.setup()

        self.performance = 0
        self.fitness = 0
        self.last_performance = 0
        self.last_fitness = 0

        self.run: Run = wandb.init(
            project=project,
            group=group,
            job_type="learner",
            config={
                "seed": self.seed,
                "progenitor": prog,
            },
        )

    def calculate_reward(self, outputs: Array) -> float:
        avg_fitness = self.behavior.grade(outputs).fitness / len(self.behavior.history)
        fitness = self.behavior.history[-1].fitness
        self.performance = (fitness - avg_fitness) / self.behavior.dt

        old = self.performances.popleft()
        self.performances.append(self.performance)
        self.total_performance += self.performance - old

        reward = self.performance - self.total_performance / self.count
        return reward * self.behavior.dt

    def calculate_displacement(self) -> float:
        flat_center: np.ndarray = self.rlctrnn.center
        flat_weights: np.ndarray = self.progenitor.weights
        return np.sqrt(np.sum(np.power(flat_center + -flat_weights, 2)))

    def iter(self):
        self.voltages = self.rlctrnn.step(self.voltages)
        outputs = self.rlctrnn.ctrnn.get_output(self.voltages)
        reward = self.calculate_reward(outputs)

        data = {}
        data["fitness"] = self.behavior.total.fitness
        data["performance"] = self.total_performance / self.count
        data["reward"] = reward
        data["flux"] = self.rlctrnn.flux
        data["time"] = self.time
        data["displacement"] = self.calculate_displacement()
        data["distance"] = self.rlctrnn.distance
        for y in range(self.rlctrnn.ctrnn.size):
            # data[f"neuron.{y}"] = outputs[y]
            for x in range(self.rlctrnn.ctrnn.size):
                data[f"weight.{x}.{y}"] = self.rlctrnn.center[x, y]

        self.run.log(data)
        self.rlctrnn.update(reward)
        self.time += self.behavior.dt


def main():
    ctrnn = Ctrnn.from_dict(
        {
            "time_constants": {0: 1.0, 1: 1.0},
            "biases": {0: -9.734175783747375, 1: 5.135667768885297},
            "weights": {
                0: {0: 5.725272596164523, 1: -16.0},
                1: {0: 13.833469092896578, 1: 0.5880424886097462},
            },
        }
    )
    learner = Learner("reward", "", ctrnn, seed=randint(1, 10000))

    while learner.time < 600:
        learner.iter()
    learner.run.finish()


if __name__ == "__main__":
    threads = []
    for i in range(10):
        threads.append(Process(target=main, args=()))
    for _, p in enumerate(threads):
        p.start()
    for _, p in enumerate(threads):
        p.join()
