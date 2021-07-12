import wandb.sdk

class Run(wandb.sdk.wandb_run.Run):
    summary: wandb.sdk.wandb_summary.Summary
    history: wandb.sdk.wandb_history.History
