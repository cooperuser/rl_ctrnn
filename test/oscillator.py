from numpy import random
import wandb
from rl_ctrnn.ranges import CtrnnRanges
import unittest
from rl_ctrnn.ctrnn import Ctrnn
from evaluator.oscillator import Oscillator

def rand(n: float = 1):
    return n * (random.random() * 2 - 1)

class Tests(unittest.TestCase):
    def test_known(self):
        for i in range(1):
            wandb.init(entity="ampersand", project="beep")
            ctrnn = Ctrnn(2)
            ctrnn.set_bias(0, -2.75 + rand(0.05))
            ctrnn.set_bias(1, -1.75 + rand(0.05))
            ctrnn.set_weight(0, 0, 4.5 + rand(0.25))
            ctrnn.set_weight(1, 0, 1.0 + rand(0.25))
            ctrnn.set_weight(0, 1, -1.0 + rand(0.25))
            ctrnn.set_weight(1, 1, 4.5 + rand(0.25))
            # ctrnn.randomize(CtrnnRanges())
            eval = Oscillator(ctrnn)
            for j in range(3000):
                eval._step()
                if j >= 2500:
                    outputs = eval.ctrnn.get_output(eval.voltages)
                    step = j - 2500
                    wandb.log({
                        "a": outputs[0],
                        "b": outputs[1],
                        "fitness": eval.fitness / (eval.ctrnn.size + step),
                        "avg_a": eval.report.averages[0] / (step + 1),
                        "avg_b": eval.report.averages[0] / (step + 1),
                    })
            report = eval.generate_report()

            wandb.finish()
        # for n in range(ctrnn.size):
        #     self.assertAlmostEqual(report.ranges[n].min, 0.1865, 4)
        #     self.assertAlmostEqual(report.ranges[n].max, 0.8135, 4)
        #     self.assertAlmostEqual(report.averages[n] / 10, 0.05, 2)
        #     self.assertGreaterEqual(report.fitness, 0.043)
        #     self.assertTrue(report.beers_metric)
