"""Generate some interesting figures."""

from functools import partial
import logging
import os
import re
from typing import List
from matplotlib.artist import Artist
from matplotlib.axis import Axis
from matplotlib import animation as ani
import click
import matplotlib.pyplot as plt

fmt_ = "[%(asctime)s] %(levelname)s %(message)s"
dft_ = "%-m/%-d/%y %H:%M:%S"
logging.basicConfig(level="INFO", format=fmt_, datefmt=dft_)


@click.command()
@click.argument("logfile", type=click.Path(exists=True, dir_okay=False))
def main(logfile: str) -> None:
    plot_losses(logfile)
    plot_texts(os.path.dirname(logfile))


def plot_texts(dir: str) -> None:
    # These parameters are set from poking around
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.1, 0.9)
    ax.set_axis_off()

    texts = sorted(
        os.path.join(dir, file)
        for file in os.listdir(dir)
        if file.startswith("generated_text")
    )

    def update(path: str, ax: Axis) -> List[Artist]:
        """Redraw the text in the image."""
        iters = int(os.path.basename(path).split(".")[0].split("_")[-1])
        with open(path) as f:
            lines = f.readlines()

        ax.texts.clear()

        artists = []
        for i, line in enumerate(lines[:20]):
            artists.append(
                ax.text(
                    0,
                    1 - i / 20,
                    line.rstrip()[:100],
                    {"size": 8},
                    transform=ax.transAxes,
                    clip_on=True,
                    animated=True,
                )
            )

        artists += [
            ax.text(
                0,
                0,
                f"Iteration {iters}",
                {"color": "red"},
                transform=ax.transAxes,
            )
        ]
        return artists

    writer = ani.PillowWriter(fps=1 / 2)
    a = ani.FuncAnimation(fig, partial(update, ax=ax), texts)
    a.save(os.path.join(dir, "texts.gif"), writer=writer, dpi=300)


def plot_losses(logfile: str) -> None:
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
