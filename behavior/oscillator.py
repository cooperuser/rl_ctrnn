from collections import deque
from behavior import Behavior, Grade
import numpy as np


class OscillatorGrade(Grade):
    fitness: float = 0.0

    def __add__(self, other):
        g = OscillatorGrade()
        g.fitness = self.fitness + other.fitness
        return g

    def __sub__(self, other):
        g = OscillatorGrade()
        g.fitness = self.fitness - other.fitness
        return g


class Oscillator(Behavior):
    history: deque[OscillatorGrade]

    def setup(self):
        count = int(self.durations[1] / self.dt)
        self.history = deque([OscillatorGrade() for _ in range(count)])
        self.total: OscillatorGrade = OscillatorGrade()
        self.last = np.zeros(self.size)
        pass

    def grade(self, state: np.ndarray) -> OscillatorGrade:
        grade = OscillatorGrade()
        grade.fitness = np.sum(np.abs(state + -self.last)) / self.size
        self.last = state
        old = self.history.popleft()
        self.history.append(grade)
        self.total += grade - old
        return self.total
