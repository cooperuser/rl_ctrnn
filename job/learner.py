from collections import deque
from multiprocessing.context import Process
from random import randint
from rl_ctrnn.rl_ctrnn import RLCtrnn
from util.run import Run
from behavior.oscillator import Oscillator
from rl_ctrnn.ctrnn import Array, Ctrnn
import numpy as np
import wandb


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
        self.behavior = Oscillator(size=self.progenitor.size)
        prog = Ctrnn.to_dict(ctrnn)
        self.rlctrnn = RLCtrnn(Ctrnn.from_dict(prog), self.seed)
        self.voltages = self.rlctrnn.ctrnn.make_instance()
        self.behavior.setup(self.rlctrnn.ctrnn.get_output(self.voltages))

        avg_count = self.behavior.avg_count
        self.performances = deque([0.0 for _ in range(20)])
        self.rewards = deque([0.0 for _ in range(avg_count)])
        self.fitness = 0
        self.total_performance = 0
        self.total_reward = 0
        self.window = self.behavior.window

        self.performance = 0
        self.last_performance = 0
        self.avg_performance = 0
        self.fitness = 0
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
        self.behavior.grade(outputs)  # - 0.1 * self.behavior.dt
        self.performance = self.behavior.history[-1] - self.behavior.history[0]

        old = self.performances.popleft()
        self.performances.append(self.performance)
        self.avg_performance += (self.performance - old) / len(self.performances)

        return self.avg_performance

    def calculate_displacement(self) -> float:
        flat_center: np.ndarray = self.rlctrnn.center
        flat_weights: np.ndarray = self.progenitor.weights
        return np.sqrt(np.sum(np.power(flat_center + -flat_weights, 2)))

    def iter(self, log=True):
        self.voltages = self.rlctrnn.step(self.voltages)
        outputs = self.rlctrnn.ctrnn.get_output(self.voltages)
        reward = self.calculate_reward(outputs)

        if log:
            data = {}
            data["fitness"] = self.behavior.fitness
            data["performance"] = self.performance
            data["reward"] = reward
            data["flux"] = self.rlctrnn.flux
            data["Time"] = self.behavior.time
            data["displacement"] = self.calculate_displacement()
            data["distance"] = self.rlctrnn.distance
            for y in range(self.rlctrnn.ctrnn.size):
                for x in range(self.rlctrnn.ctrnn.size):
                    data[f"weight.{x}.{y}"] = self.rlctrnn.center[x, y]
            self.run.log(data)

            self.rlctrnn.update(reward)


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
    learner = Learner("fit-test", "", ctrnn, seed=randint(1, 10000))

    # while learner.behavior.time < learner.behavior.window:
    #     learner.iter(False)
    # learner.behavior.time = 0

    while learner.behavior.time < learner.behavior.duration:
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
