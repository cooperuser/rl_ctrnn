from evaluator.oscillator import Oscillator
from typing import Dict, Type, List

import wandb

from rl_ctrnn.ctrnn import Array, Ctrnn
from rl_ctrnn.ranges import CtrnnRanges

# class RandomSampling(object):
#     def __init__(self, ranges: CtrnnRanges or None = None):
#         self.ranges = ranges
#         self.ctrnn = Ctrnn

CtrnnSettings: Type = dict[str, float or dict[str, float]]
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

def ctrnn_to_json(ctrnn: Ctrnn):
    data = {"weights": {}}
    for i in range(ctrnn.size):
        l = ALPHABET[i]
        data["bias_" + l] = ctrnn.biases[i]
        data["time_constant_" + l] = ctrnn.time_constants[i]
        for j, v in enumerate(ctrnn.weights[i]):
            data["weights"][l + "_to_" + ALPHABET[j]] = v
    return data

def voltage_to_json(voltages: Array):
    data = {}
    for i in range(len(voltages)):
        data[ALPHABET[i]] = voltages[i]
    return data

if __name__ == "__main__":
    ranges = CtrnnRanges()
    ranges.set_time_constant_range(1, 1)
    ctrnn = Ctrnn()
    for n in range(24000):
        print(24000 - n)
        ctrnn.randomize(ranges)
        eval = Oscillator(ctrnn, time_step=0.01)
        wandb.init(entity="ampersand", project="domain", config={
            "group": "random_sampler_0",
            "ctrnn": ctrnn_to_json(ctrnn)
        })
        for i in range(eval._transient_steps):
            eval._step()
            wandb.log({
                "neurons": voltage_to_json(ctrnn.get_output(eval.voltages))
            })
        for i in range(eval._evaluation_steps):
            eval._step()
            wandb.log({
                "neurons": voltage_to_json(ctrnn.get_output(eval.voltages)),
                "fitness": eval.fitness / eval._evaluation_steps,
                "beers_metric": eval.beers_metric
            })

        wandb.finish()
    # r.start()
    # for _ in range(10):
    #     r.evaluate()
    # r.finish()
