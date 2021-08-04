from rl_ctrnn.rl_ctrnn import RLCtrnn
import wandb
from behavior.oscillator import Oscillator
from rl_ctrnn.ctrnn import Ctrnn
from multiprocessing import Process
from time import sleep
import numpy as np

PROJECT = "rl_rule"
ITERATIONS = 10000
CTRNN = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: -9.734175783747375, 1: 5.135667768885297},
    "weights": {
        0: {0: 5.725272596164523, 1: -16.0},
        1: {0: 13.833469092896578, 1: 0.5880424886097462},
    },
}


def learner():
    seed = int(np.random.random() * 10000)
    run = wandb.init(project="rl_rule", config={"seed": seed})
    l = RLCtrnn(Ctrnn.from_dict(CTRNN), seed)
    behavior = Oscillator(2)
    behavior.setup()
    voltages = l.ctrnn.make_instance()
    last = 0
    try:
        for _ in range(10000):
            voltages = l.step(voltages)
            outputs = l.ctrnn.get_output(voltages)
            fitness = behavior.grade(outputs)
            d = behavior.size * behavior.durations[1] * behavior.dt
            reward = (fitness.fitness - last) - behavior.dt
            last = fitness.fitness
            data = {"fitness": fitness.fitness / d, "reward": reward}
            data["a"] = outputs[0]
            data["b"] = outputs[1]
            data["flux"] = l.flux
            for y in range(l.ctrnn.size):
                for x in range(l.ctrnn.size):
                    data[f"center.{x}.{y}"] = l.center[x, y]
                    data[f"weight.{x}.{y}"] = l.ctrnn.weights[x, y]
            run.log(data)
            l.update(reward)
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()


def main():
    run = wandb.init(project="fit-test")
    ctrnn = Ctrnn.from_dict(CTRNN)
    behavior = Oscillator(2)
    behavior.setup()
    voltages = ctrnn.make_instance()
    time = 0
    try:
        while time < 300:
            voltages = ctrnn.step(0.05, voltages)
            outputs = ctrnn.get_output(voltages)
            fitness = behavior.grade(outputs)
            d = behavior.size * behavior.durations[1] * 0.05
            run.log({"progenitor": fitness.fitness / d, "time": time})
            time += 0.05
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()


if __name__ == "__main__":
    learner()
    # threads = []
    # for i in range(1):
    #     threads.append(Process(target=learner, args=()))
    # for _, p in enumerate(threads):
    #     p.start()
    # for _, p in enumerate(threads):
    #     p.join()
