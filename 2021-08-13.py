from multiprocessing.context import Process
from random import randint
import re
import wandb, json
from util.run import Run
from rl_ctrnn.ctrnn import Ctrnn

THREAD_COUNT = 10
FITNESS = 0.5656714076031992
DURATION = 360

"nn2_seed-1_MS-0.2_GOAL_PRCT-0.1_ACTUAL-0.087.json"
PATTERN = r"nn2_seed-(\d+)_MS-(\d.\d+)_GOAL_PRCT-(\d.\d+)_ACTUAL-(\d.\d+).json"


def read(path) -> tuple[Ctrnn, tuple]:
    with open(path, 'r') as file:
        obj = json.loads(file.read())
        size = obj["size"]
        ctrnn = Ctrnn(size)
        for i in range(size):
            ctrnn.biases[i] = obj["biases"][i]
            ctrnn.time_constants[i] = obj["time_constants"][i]
            for j in range(size):
                ctrnn.weights[i, j] = obj["inner_weights"][i][j]
    data = (0, 0.0, 0.0, 0.0)
    if m := re.match(PATTERN, path):
        data = m.groups()
    return (ctrnn, data)


def init_run(
    group: int,
    ctrnn: Ctrnn,
    mutation: float,
    target: float,
    actual: float
) -> Run:
    return wandb.init(
        project="perturbed",
        group=group,
        config={
            "seed": group,
            "mutation": mutation,
            "percent": target,
            "actual": actual
        },
    )


# def part_a(raw):
#     ctrnn = parent.clone(self.mutation, self.rng)
#     voltages = ctrnn.make_instance()
#     behavior = self.new_behavior(ctrnn.get_output(voltages))
#     while behavior.time < behavior.duration:
#         voltages = ctrnn.step(self.dt, voltages)
#         behavior.grade(ctrnn.get_output(voltages))
#     return ctrnn, behavior.fitness
#     progenitor = Ctrnn.from_dict(PROGENITOR)
#     run = init_run("a3", seed, mutation=mutation)
#     c = NClimber(progenitor, seed=seed, duration=10, samples=1, mutation=mutation)
#     c.setup()
#     a = 0
#     run.log({"Time": a, "fitness": c.attempts[c.best][1]})
#     while a < DURATION:
#         c.single_step()
#         a += int(c.duration * c.samples)
#         data = {"Time": a, "fitness": c.attempts[c.best][1]}
#         ctrnn = c.attempts[c.best][0]
#         for y in range(ctrnn.size):
#             for x in range(ctrnn.size):
#                 data[f"weight.{x}.{y}"] = ctrnn.weights[x, y]
#         run.log(data)
#     run.finish()


if __name__ == "__main__":
    n = 9
    for t in range(n):
        for mut in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            print(t / n, mut)
            threads = []
            for _ in range(10):
                seed = randint(1, 100000)
                # threads.append(Process(target=part_a, args=(mut, seed,)))
            for _, p in enumerate(threads):
                p.start()
            for _, p in enumerate(threads):
                p.join()
