# from util.run import Run
from evaluator.oscillator import Oscillator
from typing import List
from util.run import Run
from rl_ctrnn.ctrnn import Ctrnn
import wandb


def old_to_new(c) -> dict:
    ctrnn = c["ctrnn"]
    weights = ctrnn["weights"]
    return {
        "biases": {0: ctrnn["bias_a"], 1: ctrnn["bias_b"]},
        "time_constants": {0: ctrnn["time_constant_a"], 1: ctrnn["time_constant_b"]},
        "weights": {
            0: {0: weights["a_to_a"], 1: weights["a_to_b"]},
            1: {0: weights["b_to_a"], 1: weights["b_to_b"]},
        },
    }


def hill_climber(run_id: str):
    api = wandb.Api()
    runs: List[Run] = api.runs(
        "ampersand/random_sampler",
        filters={"summary_metrics.beers_metric": {"$gte": 0.05}},
        order="+summary_metrics.fitness",
    )

    for i in range(10):
        med = runs[i]

        config = old_to_new(med.config)
        ctrnn = Ctrnn.from_dict(config)
        o = Oscillator(ctrnn)
        changes = o.run()
        run: Run = wandb.init(
            project="rl_ctrnn",
            name=med.name,
            group=med.name,
            job_type="progenitor",
            config={"ctrnn": config, "generation": 0, "attempt": 0},
        )
        run.tags = ("record",)
        for step, change in changes:
            wandb.log(change, step=step)
        wandb.finish()


if __name__ == "__main__":
    hill_climber("w5i8u8qy")
