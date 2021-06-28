import unittest
from rl_ctrnn.ctrnn import Ctrnn
from evaluator.oscillator import Oscillator

class Tests(unittest.TestCase):
    def test_known(self):
        ctrnn = Ctrnn(2)
        ctrnn.set_bias(0, -2.75)
        ctrnn.set_bias(1, -1.75)
        ctrnn.set_weight(0, 0, 4.5)
        ctrnn.set_weight(1, 0, 1.0)
        ctrnn.set_weight(0, 1, -1.0)
        ctrnn.set_weight(1, 1, 4.5)
        eval = Oscillator(ctrnn)
        for _ in range(3000):
            eval._step()
        report = eval.generate_report()

        for n in range(ctrnn.size):
            self.assertAlmostEqual(report.ranges[n].min, 0.1865, 4)
            self.assertAlmostEqual(report.ranges[n].max, 0.8135, 4)
            self.assertAlmostEqual(report.averages[n] / 10, 0.05, 2)
            self.assertGreaterEqual(report.fitness, 0.043)
            self.assertTrue(report.beers_metric)
