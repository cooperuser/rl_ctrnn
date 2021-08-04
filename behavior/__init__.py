from collections import deque
from typing import Tuple


class Grade(object):
    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass


class Behavior(object):
    def __init__(self, size: int, durations: Tuple[int, int] = (25, 5)):
        self._time = 0.0

        self.dt = 0.05
        self.size = size
        self.durations = durations
        self.history: deque[Grade] = deque()

    def setup(self):
        pass

    def grade(self, state) -> Grade:
        return Grade()
