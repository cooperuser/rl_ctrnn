from concurrent.futures import ProcessPoolExecutor as Pool
import re
from util import get_beers_fitness
from util.run import Run
from rl_ctrnn.ctrnn import Ctrnn
import wandb, json
import os

THREAD_COUNT = 10

"nn2_seed-1_MS-0.2_GOAL_PRCT-0.1_ACTUAL-0.087.json"
PATTERN = r".*nn2_seed-(\d+)_MS-(\d.\d+)_GOAL_PRCT-(\d.\d+)_ACTUAL-(\d.\d+).json"
BEST = {}


def read(path) -> tuple[Ctrnn, tuple]:
    with open(path, "r") as file:
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
    group: int, ctrnn: Ctrnn, mutation: float, target: float, actual: float
) -> Run:
    return wandb.init(
        project="perturbed",
        group=group,
        config={
            "seed": group,
            "mutation": mutation,
            "percent": target,
            "actual": actual,
        },
    )


def main(args):
    ctrnn, seed, mutation_size, goal, actual, (parent, original) = args
    fitness = get_beers_fitness(ctrnn)
    wandb.init(
        project="perturbed",
        group=seed,
        job_type=mutation_size,
        config={
            "seed": seed,
            "mutation_size": mutation_size,
            "target_percent": goal,
            "actual_percent": fitness / original,
            "fitness": fitness,
            "ctrnn": Ctrnn.to_dict(ctrnn),
            "parent": Ctrnn.to_dict(parent),
            "parent_fitness": original
        }
    )
    wandb.finish()


if __name__ == "__main__":
    best = os.scandir("/home/ampersand/ghq/github.com/jasonayoder/ctrnn_bio_rl/ctrnn_configurations/solutions/best_osc_tournaments-5k_beer/stepsize-0.01/nnsize-2")
    for file in best:
        ctrnn, _ = read(file.path)
        if m := re.match(r".*_seed-(\d+).*", file.path):
            seed = m.group(1)
            fitness = get_beers_fitness(ctrnn)
            BEST[seed] = (ctrnn, fitness)

    perturbed = os.scandir(
        "/home/ampersand/ghq/github.com/jasonayoder/ctrnn_bio_rl/ctrnn_configurations/perturbed_solutions/best_osc_tournaments-5k_beer/stepsize-0.01/nnsize-2"
    )
    todo = []
    for file in perturbed:
        ctrnn, (seed, mutation_size, goal, actual) = read(file.path)
        todo.append((ctrnn, seed, mutation_size, goal, actual, BEST[seed]))

    p = Pool(THREAD_COUNT)
    p.map(main, todo)

    # print(Ctrnn.to_dict(ctrnn), seed)
    # n = 9
    # for t in range(n):
    #     for mut in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    #         print(t / n, mut)
    #         threads = []
    #         for _ in range(10):
    #             seed = randint(1, 100000)
    #             # threads.append(Process(target=part_a, args=(mut, seed,)))
    #         for _, p in enumerate(threads):
    #             p.start()
    #         for _, p in enumerate(threads):
    #             p.join()
