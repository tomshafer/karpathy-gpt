"""BigramLM from Anrej Karpathy's GPT lecture."""

from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

OTensor = Optional[Tensor]

RANDOM_SEED = 114185


class BigramLM(nn.Module):
    """Predict next char given only the previous."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: Tensor, y: OTensor = None) -> tuple[Tensor, OTensor]:
        """Make a prediction of the next token given the current."""
        logits = self.embeddings(x)  # shape: (batch, time, vocab)
        if y is None:
            return logits, None
        b, t, c = logits.shape
        logits = logits.view(b * t, c)
        loss = F.cross_entropy(logits, y.view(b * t))
        return logits, loss

    def generate(self, x: Tensor, tokens: int = 1) -> Tensor:
        """Take an input (B, T) and sample a new token."""
        for _ in range(tokens):
            # We feed the whole context for generality, though
            # the BigramLM only uses the final token.
            logits, _ = self(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            newx = torch.multinomial(probs, 1)
            x = torch.cat((x, newx), -1)
        return x


def batch(t: torch.Tensor, batch_size: int, block_size: int) -> tuple[Tensor, Tensor]:
    ix = torch.randint(len(t) - block_size, (batch_size,))
    x = torch.stack([t[i : i + block_size] for i in ix])
    y = torch.stack([t[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def estimate_loss(model: nn.Module, trn: Tensor, tst: Tensor, args: Namespace) -> None:
    model.eval()

    data = {"train": trn, "test": tst}
    for part in data.keys():
        losses = torch.zeros(args.eval_size)
        for k in range(args.eval_size):
            x, y = batch(data[part], args.batch_size, args.context_size)
            _, losses[k] = model(x, y)
        print(f"loss({part}) = {losses.mean():.7f}", end="  ")
    print()

    model.train()


def read_text(path: str) -> str:
    with open(path) as f:
        text = f.read()
    print(f"Input length = {len(text):,d} characters")
    return text


def tokenize(text: str) -> list[str]:
    return sorted(set(text))


def encode(text: str, vocab: list[str]) -> list[int]:
    stoi = {c: i for i, c in enumerate(vocab)}
    return [stoi[c] for c in text]


def decode(ints: list[int], vocab: list[str]) -> str:
    itos = {i: c for i, c in enumerate(vocab)}
    return "".join([itos[n] for n in ints])


def cli() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--batch-size", "-b", type=int, default=32)
    p.add_argument("--context-size", "-c", type=int, default=8)
    p.add_argument("--max-iters", "-i", type=int, default=3_000)
    p.add_argument("--learning-rate", "-r", type=float, default=1e-2)
    p.add_argument("--eval-interval", "-e", type=int, default=300)
    p.add_argument("--eval-size", "-s", type=int, default=200)
    p.add_argument("--test-split", "-x", type=float, default=0.1)
    p.add_argument("FILE", type=str)
    return p.parse_args()


def train_bigram(args: Namespace) -> None:
    torch.manual_seed(RANDOM_SEED)

    text = read_text(args.FILE)
    vocab = tokenize(text)
    data = torch.tensor(encode(text, vocab), dtype=torch.long)

    nsplit = int((1 - args.test_split) * len(data))
    train, test = data[:nsplit], data[nsplit:]
    print(f"|Train| = {len(train):,}, |Test| = {len(test):,}")

    model = BigramLM(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for iter in range(args.max_iters):
        xb, yb = batch(train, batch_size=args.batch_size, block_size=args.context_size)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            estimate_loss(model, train, test, args)

    # Show an output
    print()
    print("Example output:")

    seed = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(seed, 500)[0].tolist(), vocab))


if __name__ == "__main__":
    train_bigram(cli())
