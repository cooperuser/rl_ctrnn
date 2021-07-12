from typing import List
from util.run import Run
import wandb

api = wandb.Api()


def get_group(index: int = 0) -> str:
    progenitors: List[Run] = api.runs(
        "ampersand/rl_ctrnn",
        filters={"jobType": "progenitor"},
        order="-summary_metrics.fitness",
    )
    return progenitors[index].name or ""


def get_parent(group: str, job_type: str, order=None) -> Run:
    family = api.runs(
        "ampersand/rl_ctrnn",
        order=order,
        filters={
            "group": group,
            "$or": [{"jobType": job_type}, {"jobType": "progenitor"}],
        },
    )
    return family[0]


def get_attempt(group: str, job_type: str) -> int:
    family = api.runs(
        "ampersand/rl_ctrnn",
        filters={
            "group": group,
            "$or": [{"jobType": job_type}, {"jobType": "progenitor"}],
        },
    )
    return len(family)


def get_best(group, job_type: str) -> float:
    family = api.runs(
        "ampersand/rl_ctrnn",
        order="-summary_metrics.fitness",
        filters={
            "group": group,
            "$or": [{"jobType": job_type}, {"jobType": "progenitor"}],
        },
    )
    return family[0].summary["fitness"]
