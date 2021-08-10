from collections import deque
from behavior import Behavior
import numpy as np


class Oscillator(Behavior):
    history: deque[float]

    def setup(self, state: np.ndarray):
        self.avg_count = int(self.window / self.dt)
        self.history = deque([0.0 for _ in range(self.avg_count)])
        self.fitness: float = 0.0
        self.last = state

    def grade(self, state: np.ndarray) -> float:
        grade = np.sum(np.abs(state + -self.last)) / self.size
        self.last = state
        old = self.history.popleft()
        self.history.append(grade)
        change = (grade - old) / self.window
        self.fitness += change
        self.time += self.dt
        return grade
