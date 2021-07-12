from job.climber import hill_climber
from job.walker import random_walker
from multiprocessing.context import Process
from typing import List
import os


if __name__ == "__main__":
    cycles = 10
    thread_count = 10
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    for cycle in range(cycles):
        for job in [random_walker, hill_climber]:
            threads: List[Process] = []
            for t in range(thread_count):
                threads.append(Process(target=job, args=(t,)))
            for _, p in enumerate(threads):
                p.start()
            for _, p in enumerate(threads):
                p.join()

        c = cycle + 1
        print(f"{c * thread_count} / {cycles * thread_count}\t{c/cycles}")
