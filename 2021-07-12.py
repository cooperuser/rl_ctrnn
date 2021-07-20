from job.walker import Walker
from job.climber import Climber
from rl_ctrnn.ctrnn import Ctrnn
from multiprocessing import Process

PROJECT = "mutation-sizes"
MUTATION_SIZES = [0.01, 0.05, 0.2, 0.3]
GROUP = "a"
ITERATIONS = 1000
CTRNN = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: -9.734175783747375, 1: 5.135667768885297},
    "weights": {
        0: {0: 5.725272596164523, 1: -16.0},
        1: {0: 13.833469092896578, 1: 0.5880424886097462},
    },
}


def main(seed: int):
    ctrnn = Ctrnn.from_dict(CTRNN)
    for mut in MUTATION_SIZES:
        c = Climber(PROJECT, GROUP, ctrnn, seed, mut)
        for _ in range(ITERATIONS):
            c.iter()
        c.run.finish()
        w = Walker(PROJECT, GROUP, ctrnn, seed, mut)
        for _ in range(ITERATIONS):
            w.iter()
        w.run.finish()


if __name__ == "__main__":
    threads = []
    for i in range(10):
        threads.append(Process(target=main, args=(i,)))
    for _, p in enumerate(threads):
        p.start()
    for _, p in enumerate(threads):
        p.join()
