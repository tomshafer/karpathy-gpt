from __future__ import annotations

import logging
import time
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module

from bigram import BigramLM
from transformer import TransformerLM

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%-m/%-d/%y %H:%M:%S",
)

log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CLI:
    file: str
    batch_size: int
    context_size: int
    embedding_size: int
    num_heads: int
    num_blocks: int
    dropout: float
    iter: int
    model: str
    learning_rate: float
    cuda: bool
    eval_cadence: int
    eval_samples: int

    @classmethod
    def from_args(cls):
        p = ArgumentParser()
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--context-size", type=int, default=256)
        p.add_argument("--embedding-size", type=int, default=384)
        p.add_argument("--num-heads", type=int, default=6)
        p.add_argument("--num-blocks", type=int, default=6)
        p.add_argument("--dropout", type=float, default=0.2)
        p.add_argument("--model", type=str, default="Transformer")
        p.add_argument("--iter", type=int, default=5000)
        p.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
        p.add_argument("--cuda", action="store_true")
        p.add_argument("--eval-cadence", type=int, default=500)
        p.add_argument("--eval-samples", type=int, default=200)
        p.add_argument("file", metavar="FILE", type=str, help="source text file")
        return cls(**vars(p.parse_args()))


def train_model() -> None:
    args = CLI.from_args()
    log.info(args)

    with open(args.file) as f:
        text = f.read()

    vocab = sorted(set(text))
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    log.info(f"Vocab size = {len(vocab):,}")

    nsplit = int(0.9 * len(data))
    trn, val = data[:nsplit], data[nsplit:]
    log.info(f"Train size = {len(trn):,}  Val size = {len(val):,}")

    if args.model.lower().startswith("bigram"):
        model: Module = BigramLM(len(vocab))
    elif args.model.lower().startswith("transformer"):
        model = TransformerLM(
            len(vocab),
            args.context_size,
            args.embedding_size,
            args.embedding_size // args.num_heads,
            args.num_blocks,
        )
    else:
        raise ValueError(f"unrecognized model type: '{args.model}'")

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model has {np_:,} parameters")

    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    with timer():
        for it in range(args.iter):
            x, y = random_batch(data, args.context_size, args.batch_size)
            _, losses = model(x, y)

            optim.zero_grad(set_to_none=True)
            losses.backward()
            optim.step()

            # Estimate the loss regularly
            if (it + 1) % args.eval_cadence == 0:
                msg = f"Iter {it+1:05d}: "
                model.eval()
                ddict = {"train": trn, "val": val}
                for part in ddict:
                    losses = torch.zeros(args.eval_samples)
                    for k in range(args.eval_samples):
                        bargs_ = (ddict[part], args.context_size, args.batch_size)
                        x, y = random_batch(*bargs_)
                        _, losses[k] = model(x, y)
                    msg += f"{part.capitalize()} loss = {losses.mean():.5f}  "
                log.info(msg.rstrip())
                model.train()

    # Generate some text
    prompt = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(prompt, 2000)[0].tolist(), vocab))  # type: ignore


if __name__ == "__main__":
    train_model()
