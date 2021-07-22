import wandb
from job.climber import Climber
from rl_ctrnn.ctrnn import Ctrnn

ITERATIONS = 1000
CTRNN = {
    "time_constants": {0: 1.0, 1: 1.0},
    "biases": {0: -9.734175783747375, 1: 5.135667768885297},
    "weights": {
        0: {0: 5.725272596164523, 1: -16.0},
        1: {0: 13.833469092896578, 1: 0.5880424886097462},
    },
}


def make_button(ctrnn: Ctrnn) -> wandb.Html:
    json = str(Ctrnn.to_dict(ctrnn))
    # button = f"<a href='{json}'><button>Open in Visualizer</button></a>"
    button = """
        <button style='width: 45%'>Copy parameters</button>
        <button style='width: 45%'>Open in visualizer</button>
        <iframe
            style='width: 100%; border: 0;'
            src='https://cooperuser.dev'>
        </iframe>"""
    return wandb.Html(button)

if __name__ == "__main__":
    ctrnn = Ctrnn.from_dict(CTRNN)
    w = Climber("table-test", "", ctrnn, 1, 0.05)
    # for _ in range(ITERATIONS):
    #     w.iter()
    best = w.attempts[w.best].ctrnn
    w.run.log({"best": make_button(best)})
    w.run.finish()
