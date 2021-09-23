from util.run import Run
from rl_ctrnn.ctrnn import Ctrnn
from typing import List
import wandb


class Perturbed(object):
    ctrnn: Ctrnn
    progenitor: Ctrnn
    progenitor_seed: int
    progenitor_mutation_size: float
    progenitor_fitness: float
    target_percent: float
    fitness: float
    actual_percent: float

    def __init__(self, run: Run) -> None:
        config = run.config
        self.ctrnn = Ctrnn.from_dict_legacy(config["ctrnn"])
        self.progenitor = Ctrnn.from_dict_legacy(config["parent"])
        self.progenitor_seed = int(config["seed"])
        self.progenitor_mutation_size = float(config["mutation_size"])
        self.progenitor_fitness = float(config["parent_fitness"])
        self.target_percent = float(config["target_percent"])
        self.fitness = float(config["fitness"])
        self.actual_percent = float(config["actual_percent"])

    def __repr__(self) -> str:
        return str(self.__dict__)


def get_perturbed_runs() -> List[Perturbed]:
    api = wandb.Api()
    runs: List[Run] = api.runs(path="ampersand/perturbed")
    return list(map(Perturbed, runs))
