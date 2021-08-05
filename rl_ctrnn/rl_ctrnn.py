from collections import deque
from rl_ctrnn import ranges
from typing import Generic, TypeVar
from typing_extensions import TypeAlias
from time import sleep

import wandb
from rl_ctrnn.ctrnn import Ctrnn
import numpy as np
import numpy.typing as npt

Array: TypeAlias = np.ndarray
Datum: TypeAlias = npt.NDArray[np.float32]


class Clamp(object):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max


class Bounds(object):
    def __init__(self) -> None:
        self.flux = Clamp(0, 10)
        self.center = Clamp(-16, 16)
        self.period = Clamp(2, 10)


class RLCtrnn(object):
    def __init__(self, ctrnn: Ctrnn, seed):
        self.ctrnn = ctrnn
        self.bounds = Bounds()
        self.rng = np.random.default_rng(seed)

        shape = (ctrnn.size, ctrnn.size)
        self.flux: float = 5
        self.center = self.ctrnn.weights
        self.period = np.random.uniform(
            self.bounds.period.min, self.bounds.period.max, size=shape
        )
        self.time: Datum = np.zeros(shape)

        self.flux_conv_rate = 0.1
        self.max_flux_amp = 10
        self.learn_rate = 1.0
        self.distance = 0.0

    def update(self, reward: float = 0):
        self.flux -= self.flux_conv_rate * self.max_flux_amp * reward
        self.flux = np.clip(self.flux, 0, self.max_flux_amp)
        displacement = self.flux * np.sin(2 * np.pi * self.time / self.period)
        center = self.center + self.learn_rate * displacement * reward
        center = center.clip(self.bounds.center.min, self.bounds.center.max)
        self.distance += np.sqrt(np.sum(np.power(center + -self.center, 2)))
        self.center = center

    def step(self, voltages: Array = np.empty(0), dt: float = 0.05):
        size = self.ctrnn.size
        self.time += dt
        time = self.time.flat
        period = self.period.flat
        stale = filter(lambda i: time[i] > period[i], range(size ** 2))
        for i in stale:
            self.time.put(i, 0)
            p = self.bounds.period
            self.period.put(i, self.rng.uniform(p.min, p.max))
        displacement = self.flux * np.sin(2 * np.pi * self.time / self.period)
        self.ctrnn.weights = self.center + displacement
        return self.ctrnn.step(dt, voltages)


if __name__ == "__main__":
    run = wandb.init(project="rl_rule")
    seed = int(np.random.random() * 10000)
    r = RLCtrnn(Ctrnn(2), seed)
    r.ctrnn.set_bias(0, -2.75)
    r.ctrnn.set_bias(1, -1.75)
    voltages = r.ctrnn.make_instance()
    last: Array = np.array(2)
    duration = int(5 / 0.05)
    history = deque([0 for _ in range(duration)])
    time = 0
    try:
        for _ in range(10000):
            voltages = r.step(voltages)
            outputs: Array = r.ctrnn.get_output(voltages)
            history.popleft()
            newest = np.sum(np.abs(outputs + -last))
            history.append(newest)
            last = outputs

            fitness = np.mean(history)
            reward = newest - fitness
            # print()
            # print(outputs)
            # print(fitness, reward)
            d = {}
            d["a"] = outputs[0]
            d["b"] = outputs[1]
            d["fitness"] = np.sum(history) / (2 * duration * 0.05)
            d["reward"] = reward
            d["flux"] = r.flux
            for y in range(r.ctrnn.size):
                for x in range(r.ctrnn.size):
                    d[f"center.{x}.{y}"] = r.center[x, y]
                    d[f"weight.{x}.{y}"] = r.ctrnn.weights[x, y]
            run.log(d)
            r.update(reward)
            time += 0.05
    except KeyboardInterrupt:
        exit(0)
    finally:
        run.finish()
