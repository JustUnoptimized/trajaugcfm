import itertools

import jaxtyping as jt
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        d_in:  int,
        d_out: int,
        w:     int=64,
        h:     int=2
    ) -> None:
        super().__init__()
        # self.d_in = d_in
        # self.d_out = d_out
        # self.w = w
        # self.h = h
        act = nn.SELU()  ## will get registered inside nn.Sequential()
        linears = (nn.Linear(w, w) for _ in range(h))
        self.net = nn.Sequential(
            nn.Linear(d_in, w),
            act,
            *itertools.chain.from_iterable(zip(linears, itertools.repeat(act))),
            nn.Linear(w, d_out)
        )

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> jt.Real[torch.Tensor, '*batch d_out']:
        return self.net(x)


class FlowScoreMLP(nn.Module):
    def __init__(
        self,
        d_in:  int,
        d_out: int,
        w:     int=64,
        h:     int=2
    ) -> None:
        super().__init__()
        act = nn.SELU()
        layers = [nn.Linear(d_in, w), act]
        for _ in range(h):
            layers.append(nn.Linear(w, w))
            layers.append(act)
        self.trunk = nn.Sequential(*layers)
        self.head_flow = nn.Linear(w, d_out)
        self.head_score = nn.Linear(w, d_out)

    def forward(
        self,
        x: jt.Real[torch.Tensor, '*batch d_in']
    ) -> tuple[jt.Real[torch.Tensor, '*batch d_out'], jt.Real[torch.Tensor, '*batch d_out']]:
        h = self.trunk(x)
        return self.head_flow(h), self.head_score(h)


def main() -> None:
    d_in = 3
    d_out = 5
    w = 4
    d = 2
    print(d_in, d_out, w, d)
    f = MLP(d_in+1, d_out, w, d)
    print(f)

    b = 8
    x = torch.rand(b, d_in)
    t = torch.rand(b, 1)
    xt = torch.concat((t, x), dim=-1)
    y = torch.rand(b, d_out)

    print(x.shape, t.shape, xt.shape, y.shape)

    yhat = f(xt)
    print(yhat.shape)


if __name__ == '__main__':
    main()
