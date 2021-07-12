from random import randint

import wandb
from util.run import Run
import numpy as np
from evaluator.oscillator import Oscillator
from rl_ctrnn.ctrnn import Ctrnn

from job import *


def random_walker(index: int = 0):
    group = get_group(index)
    parent = get_parent(group, "walker")
    attempt = get_attempt(group, "walker")
    best = get_best(group, "walker")
    seed = randint(0, 16 ** 8)
    np.random.seed(seed)
    ctrnn = Ctrnn.from_dict(parent.config["ctrnn"], 0.1)
    o = Oscillator(ctrnn)
    changes = o.run()
    run: Run = wandb.init(
        project="rl_ctrnn",
        group=group,
        job_type="walker",
        config={
            "ctrnn": Ctrnn.to_dict(ctrnn),
            "attempt": attempt,
            "generation": parent.config["generation"] + 1,
        },
    )
    run.summary["attempt"] = attempt
    if o.fitness / 500 > best:
        run.tags = ("record",)
    for step, change in changes:
        run.log(change, step=step)
    run.log({"attempt": attempt})
    run.finish()


if __name__ == "__main__":
    random_walker(0)
    # cycles = 20
    # thread_count = 10
    # os.environ["WANDB_SILENT"] = "true"
    # os.environ["WANDB_CONSOLE"] = "off"
    # for cycle in range(cycles):
    #     for i in range(2):
    #         offset = i * 10
    #         threads: List[Process] = []
    #         for t in range(thread_count):
    #             p = Process(target=hill_climber, args=(t + offset,))
    #             threads.append(p)
    #         for _, p in enumerate(threads):
    #             p.start()
    #         for _, p in enumerate(threads):
    #             p.join()

    #     c = cycle + 1
    #     print(f"{c * thread_count} / {cycles * thread_count}\t{c/cycles}")
