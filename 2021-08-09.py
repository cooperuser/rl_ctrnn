from multiprocessing.context import Process
import wandb
from behavior.oscillator import Oscillator
from job.learner import Learner
from rl_ctrnn.ctrnn import Ctrnn
from random import randint


THREAD_COUNT = 10
PROGENITOR = {
  "time_constants": {0: 1.0, 1: 1.0},
  "biases": {0: 5.154455202973727, 1: -10.756384207938911},
  "weights": {
    0: {0: 5.352730101212875, 1: 16.0},
    1: {0: -11.915400080418113, 1: 2.7717190607157542},
  },
}


def get_frozen(ctrnn: Ctrnn) -> float:
    voltages = ctrnn.make_instance()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=300, window=50)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness

def get_log_data(m: Learner) -> dict:
    data = {"Time": m.behavior.time}
    data["fitness"] = m.behavior.fitness
    data["distance"] = m.rlctrnn.distance
    data["displacement"] = m.calculate_displacement()
    # d_b = calculate_displacement(m.rlctrnn.center, WEIGHTS_OPTIMAL)
    # data["remaining"] = d_b

    data["performance"] = m.performance
    data["reward"] = m.reward
    data["flux"] = m.rlctrnn.flux

    for y in range(m.rlctrnn.ctrnn.size):
        for x in range(m.rlctrnn.ctrnn.size):
            data[f"weight.{x}.{y}"] = m.rlctrnn.center[x, y]
    return data



def main(seed):
    progenitor = Ctrnn.from_dict(PROGENITOR)
    initial_fitness = get_frozen(progenitor)

    run = wandb.init(project="temporary", config={
        "progenitor": PROGENITOR,
        "initial_fitness": initial_fitness,
        "seed": seed,
    })

    m = Learner(progenitor, seed)
    m.behavior.dt = 0.01  # TODO: FIX EVERYWHERE
    m.behavior.duration = 100
    m.behavior.window = 10

    while m.is_running():
        m.iter()
        data = get_log_data(m)
        run.log(data)

    final_fitness = get_frozen(m.rlctrnn.ctrnn)
    run.summary["final_fitness"] = final_fitness
    run.finish()


if __name__ == "__main__":
    for _ in range(9):
        threads = []
        for _ in range(THREAD_COUNT):
            seed = randint(1, 100000)
            threads.append(Process(target=main, args=(seed,)))
        for _, p in enumerate(threads):
            p.start()
        for _, p in enumerate(threads):
            p.join()
