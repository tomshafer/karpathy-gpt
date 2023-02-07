"""Generate some interesting figures."""

import logging
import os
import re

import click
import matplotlib.pyplot as plt

fmt_ = "[%(asctime)s] %(levelname)s %(message)s"
dft_ = "%-m/%-d/%y %H:%M:%S"
logging.basicConfig(level="INFO", format=fmt_, datefmt=dft_)


@click.command()
@click.argument("logfile", type=click.Path(exists=True, dir_okay=False))
def main(logfile: str) -> None:
    RE = re.compile(r"Iter ([0-9]+).*?Train loss = ([0-9.-]+)\s+?Val loss = ([0-9.-]+)")
    with open(logfile) as f:
        losses = [RE.search(line) for line in f if "Train loss =" in line]
        data = [list(map(float, loss.groups())) for loss in filter(None, losses)]

    fig, axes = plt.subplots(ncols=2, figsize=(10, 3.5))
    for ax in axes:
        ax.plot([i for i, _, _ in data], [t for _, t, _ in data], label="Train")
        ax.plot([i for i, _, _ in data], [v for _, _, v in data], label="Val")
        ax.set_xlabel("Iteration")

    axes[0].legend()
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Linear scale")
    axes[1].set_yscale("log")
    axes[1].set_title("Logarithmic scale")
    fig.tight_layout()

    outpath = os.path.join(os.path.dirname(logfile), "loss.png")
    fig.savefig(outpath)


if __name__ == "__main__":
    main()
