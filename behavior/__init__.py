from collections import deque


class Behavior(object):
    def __init__(
        self,
        dt: float = 0.05,
        size: int = 2,
        window: float = 30.0,
        duration: float = 120.0,
    ):
        self.dt = dt
        self.size = size
        self.window = window
        self.duration = duration
        self.history: deque[float] = deque()
        self.time = 0.0

    def setup(self):
        pass

    def grade(self, state) -> float:
        return 0.0
