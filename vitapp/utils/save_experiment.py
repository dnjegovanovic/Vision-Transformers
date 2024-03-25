import json, math
from pathlib import Path
import torch


def save_checkpoint(
    experiment_name: Path, model, epoch: int, base_dir: Path = "experiments"
):
    """

    :param experiment_name: Experiment name
    :param model: torch model
    :param epoch: epoch
    :param base_dir: base dir path
    :return:
    """
    outdir = base_dir / experiment_name
    outdir.mkdir(exist_ok=True)
    cpfile = outdir / f"model_{epoch}.pt"
    torch.save(model.state_dict(), cpfile)


def save_experiment(
    experiment_name: Path,
    config,
    model,
    train_losses,
    test_losses,
    accuracies,
    base_dir: Path = "experiments",
):
    outdir = base_dir / experiment_name
    outdir.mkdir(exist_ok=True)

    # Save the config
    configfile = outdir / "config.json"
    with open(configfile, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = outdir / "metrics.json"
    with open(jsonfile, "w") as f:
        data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracies": accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def load_experiment(
    experiment_name, ViTModel, checkpoint_name="model_final.pt", base_dir="experiments"
):
    outdir = base_dir / experiment_name
    # Load the config
    configfile = outdir / "config.json"
    with open(configfile, "r") as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = outdir / "metrics.json"
    with open(jsonfile, "r") as f:
        data = json.load(f)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    accuracies = data["accuracies"]
    # Load the model
    model = ViTModel(config)
    cpfile = outdir / checkpoint_name
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies
