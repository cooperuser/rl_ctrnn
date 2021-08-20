from collections import deque
from multiprocessing.context import Process
from random import randint
from rl_ctrnn.rl_ctrnn import RLCtrnn
from behavior.oscillator import Oscillator
from rl_ctrnn.ctrnn import Array, Ctrnn
import numpy as np


class Learner(object):
    def __init__(
        self,
        ctrnn: Ctrnn,
        seed: int = 0,
        reward_ratio: float = 0.8,
        smoothness: float = 0.005,
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.reward_ratio = reward_ratio
        self.smoothness = smoothness
        self.behavior = Oscillator(size=self.progenitor.size)
        prog = Ctrnn.to_dict(ctrnn)
        self.rlctrnn = RLCtrnn(Ctrnn.from_dict(prog), self.seed)
        self.voltages = self.rlctrnn.ctrnn.make_instance()
        self.behavior.setup(self.rlctrnn.ctrnn.get_output(self.voltages))

        self.performances = deque([0.0 for _ in range(20)])
        self.window = self.behavior.window

        self.performance = 0
        self.avg_performance = 0
        self.fitness = 0

    def calculate_reward(self, outputs: Array) -> float:
        change = self.behavior.grade(outputs) - self.reward_ratio * self.behavior.dt
        reward = change - self.performance

        self.performance *= 1 - self.smoothness
        self.performance += change * self.smoothness
        return reward

    def calculate_displacement(self) -> float:
        flat_center: np.ndarray = self.rlctrnn.center
        flat_weights: np.ndarray = self.progenitor.weights
        return np.sqrt(np.sum(np.power(flat_center + -flat_weights, 2)))

    def iter(self):
        self.voltages = self.rlctrnn.step(self.voltages, 0.01)  # TODO: fix everywhere
        outputs = self.rlctrnn.ctrnn.get_output(self.voltages)
        self.reward = self.calculate_reward(outputs)
        self.rlctrnn.update(self.reward)

    def is_running(self) -> bool:
        if self.rlctrnn.flux < 0.01:
            return False
        if self.behavior.time < self.behavior.duration:
            return True
        return False


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
    learner = Learner(ctrnn, seed=randint(1, 10000))

    while learner.behavior.time < learner.behavior.window:
        learner.iter()
    learner.behavior.time = 0

    while learner.behavior.time < learner.behavior.duration:
        learner.iter()


if __name__ == "__main__":
    threads = []
    for i in range(10):
        threads.append(Process(target=main, args=()))
    for _, p in enumerate(threads):
        p.start()
    for _, p in enumerate(threads):
        p.join()
