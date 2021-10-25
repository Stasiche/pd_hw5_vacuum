import os.path

import wandb
from datetime import datetime


def wandb_init(global_config):
    cfg = global_config["wandb"]

    wandb.login()

    now = datetime.now()

    run = wandb.init(
        project=cfg["project"],
        name=f'{cfg["name"]}:{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}',
        group=cfg.get("group", None),
        notes=cfg["notes"],
        entity=cfg["entity"],
        config=global_config,
    )

    return run, wandb.config


def wandb_log(exp_dir_name, reward_mean, reward_std, metric_mean, metric_std):
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(os.path.join(exp_dir_name, "model.pth"))
    wandb.log_artifact(artifact)

    wandb.log(
        {
            "reward/mean": reward_mean,
            "reward/std": reward_std,
            "metric/mean": metric_mean,
            "metric/std": metric_std,
        },
    )
