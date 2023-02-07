"""Main training script."""

from __future__ import annotations

import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from textwrap import indent

import torch
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from bigram import BigramLM
from transformer import TransformerLM

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%-m/%-d/%y %H:%M:%S",
)

log = logging.getLogger(__name__)


def setup_logging(cfg: CN, lh: logging.Logger) -> None:
    # Add a file handler for archival
    dest = f"log-{datetime.now().isoformat().replace(':', '')}.txt"
    dest = os.path.join(cfg.RUN.OUTPUT_DIR, dest)
    lh.info(f"Logging to {dest}")

    h = logging.FileHandler(dest)
    h.setFormatter(logging.getLogger().handlers[0].formatter)
    lh.addHandler(h)


def encode(text: str, vocab: list[str]) -> list[int]:
    """Encode text from string to ints."""
    stoi = {c: i for i, c in enumerate(vocab)}
    return [stoi[c] for c in text]


def decode(ints: list[int], vocab: list[str]) -> str:
    """Decode ints to text."""
    itos = {i: c for i, c in enumerate(vocab)}
    return "".join([itos[n] for n in ints])


def random_batch(
    t: Tensor, context_size: int, batch_size: int
) -> tuple[Tensor, Tensor]:
    ix = torch.randint(len(t) - context_size, (batch_size,))
    x = torch.stack([t[i : i + context_size] for i in ix])
    y = torch.stack([t[i + 1 : i + 1 + context_size] for i in ix])
    return x, y


class timer:
    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args, **kwargs) -> None:
        end = time.perf_counter()
        log.info(f"Finshed in {end - self.start:.5f} sec")


# Model config
_C = CN()

_C.RUN = CN()
_C.RUN.DATA = None
_C.RUN.OUTPUT_DIR = "."
_C.RUN.CUDA = False

_C.MODEL = CN()
_C.MODEL.CLASS = "Transformer"
_C.MODEL.PARAMS = CN()
_C.MODEL.PARAMS.CONTEXT_SIZE = 256
_C.MODEL.PARAMS.EMBEDDING_SIZE = 384
_C.MODEL.PARAMS.NUM_HEADS = 6
_C.MODEL.PARAMS.NUM_BLOCKS = 6
_C.MODEL.PARAMS.DROPOUT_RATIO = 0.2

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_ITERS = None
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.CHECKPOINT_ITERS = None

_C.EVAL = CN()
_C.EVAL.CADENCE_ITERS = 500
_C.EVAL.NUM_SAMPLES = 200
_C.EVAL.GENERATION_ITERS = []
_C.EVAL.GENERATION_SIZE = 2000


def cfg_from_cli() -> CN:
    """Bring in configs from file."""
    p = ArgumentParser()
    p.add_argument("cfg_file", metavar="CONFIG_YAML")
    cfg = _C.clone()
    cfg.merge_from_file(p.parse_args().cfg_file)
    return cfg


def train_model() -> None:
    C = cfg_from_cli()
    setup_logging(C, log)
    log.info("Configuration:\n" + indent(str(C), 4 * " "))

    with open(C.RUN.DATA) as f:
        text = f.read()

    vocab = sorted(set(text))
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    log.info(f"Vocab size = {len(vocab):,}")

    nsplit = int(0.9 * len(data))
    trn, val = data[:nsplit], data[nsplit:]
    log.info(f"Train size = {len(trn):,}  Val size = {len(val):,}")

    if C.MODEL.CLASS.lower().startswith("bigram"):
        model: Module = BigramLM(len(vocab))
    elif C.MODEL.CLASS.lower().startswith("transformer"):
        model = TransformerLM(
            len(vocab),
            C.MODEL.PARAMS.CONTEXT_SIZE,
            C.MODEL.PARAMS.EMBEDDING_SIZE,
            C.MODEL.PARAMS.EMBEDDING_SIZE // C.MODEL.PARAMS.NUM_HEADS,
            C.MODEL.PARAMS.NUM_BLOCKS,
        )
    else:
        raise ValueError(f"unrecognized model type: '{C.MODEL.CLASS}'")

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model has {np_:,} parameters")

    DEV = torch.device("cuda" if C.RUN.CUDA else "cpu")
    log.info(f"Using device '{DEV}'")

    data = data.to(DEV)
    trn = trn.to(DEV)
    val = val.to(DEV)

    model = model.to(DEV)
    optim = torch.optim.Adam(model.parameters(), lr=C.TRAIN.LEARNING_RATE)

    with timer():
        for it in range(C.TRAIN.NUM_ITERS):
            x, y = random_batch(data, C.MODEL.PARAMS.CONTEXT_SIZE, C.TRAIN.BATCH_SIZE)
            _, losses = model(x, y)

            optim.zero_grad(set_to_none=True)
            losses.backward()
            optim.step()

            # Estimate the loss regularly
            if (it + 1) % C.EVAL.CADENCE_ITERS == 0:
                msg = f"Iter {it+1:05d}: "
                model.eval()
                ddict = {"train": trn, "val": val}
                for part in ddict:
                    losses = torch.zeros(C.EVAL.NUM_SAMPLES, device=DEV)
                    for k in range(C.EVAL.NUM_SAMPLES):
                        bargs_ = (ddict[part], C.MODEL.PARAMS.CONTEXT_SIZE, 1)
                        x, y = random_batch(*bargs_)
                        _, losses[k] = model(x, y)
                    msg += f"{part.capitalize()} loss = {losses.mean():.5f}  "
                log.info(msg.rstrip())
                model.train()

            # Checkpoint the model
            if (it + 1) % C.TRAIN.CHECKPOINT_ITERS == 0:
                log.info(f"Iter {it+1:05d}: Checkpointing model")
                torch.save(
                    obj={
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": losses.mean().item(),
                    },
                    f=os.path.join(C.RUN.OUTPUT_DIR, f"model_{it+1:06d}.pt"),
                )

            # Generate text
            if (it + 1) in C.EVAL.GENERATION_ITERS:
                log.info(f"Iter {it+1:05d}: Generating output")
                p = os.path.join(C.RUN.OUTPUT_DIR, f"generated_text_{it+1:06d}.txt")
                with open(p, "w") as f:
                    with torch.no_grad():
                        model.eval()
                        with timer():
                            prompt = torch.zeros((1, 1), dtype=torch.long, device=DEV)
                            seq = model.generate(prompt, C.EVAL.GENERATION_SIZE)  # type: ignore[operator]
                            f.write(decode(seq[0].tolist(), vocab))
                        model.train()

    # Generate some text
    prompt = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    seq = model.generate(prompt, 10 * C.EVAL.GENERATION_SIZE)  # type: ignore[operator]
    print(decode(seq[0].tolist(), vocab))


if __name__ == "__main__":
    train_model()
