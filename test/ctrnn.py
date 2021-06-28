import unittest
from rl_ctrnn.ctrnn import Ctrnn, Range

class Tests(unittest.TestCase):
    def test_creation(self):
        ctrnn = Ctrnn(2)
        self.assertEqual(ctrnn.size, 2)
        self.assertEqual(ctrnn.biases.size, 2)
        self.assertEqual(ctrnn.time_constants.size, 2)
        self.assertEqual(ctrnn.weights.size, 4)

    def test_make_instance(self):
        ctrnn = Ctrnn(2)
        v = ctrnn.make_instance()
        self.assertEqual(v.size, ctrnn.size)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

    def test_oscillation(self):
        ctrnn = Ctrnn(2)
        ctrnn.set_bias(0, -2.75)
        ctrnn.set_bias(1, -1.75)
        ctrnn.set_weight(0, 0, 4.5)
        ctrnn.set_weight(1, 0, 1.0)
        ctrnn.set_weight(0, 1, -1.0)
        ctrnn.set_weight(1, 1, 4.5)
        v = ctrnn.make_instance()
        for _ in range(70):
            v = ctrnn.step(0.1, v)
        a = 0
        b = 0
        range_a = Range(1, 0)
        range_b = Range(1, 0)
        iterations = 5000
        for _ in range(iterations):
            v = ctrnn.step(0.1, v)
            o = ctrnn.get_output(v)
            a += o[0]
            b += o[1]
            range_a.set_clamp(o[0])
            range_b.set_clamp(o[1])
        self.assertAlmostEqual(a / iterations, 0.5, 2)
        self.assertAlmostEqual(b / iterations, 0.5, 2)
        self.assertLess(range_a.max, 0.83)
        self.assertLess(range_b.max, 0.83)
        self.assertGreater(range_a.min, 0.17)
        self.assertGreater(range_b.min, 0.17)
