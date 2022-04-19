import torch
import pickle
import gzip
import numpy as np


def F1(x):
    return 0.33 * (
        torch.sin(np.pi * x[:, [0]])
        + torch.sin(2.0 * np.pi * x[:, [1]] + np.pi / 8.0)
        + x[:, [1]]
        - x[:, [2]] * x[:, [3]]
    )


def F2(x):
    """Requires 2 hidden layers."""
    return (
        torch.sin(np.pi * x[:, [0]])
        + x[:, [1]] * torch.cos(2 * np.pi * x[:, [0]] + np.pi / 4.0)
        + x[:, [2]]
        - x[:, [3]] ** 2
    ) / 3.0


def F3(x):
    """Requires 2 hidden layers."""
    return (
        (1.0 + x[:, [1]]) * torch.sin(np.pi * x[:, [0]])
        + x[:, [1]] * x[:, [2]] * x[:, [3]]
    ) / 3.0


def F4(x1, x2, x3, x4):
    """Requires 4 hidden layers."""
    y0 = 0.5 * (
        np.sin(np.pi * x1) + np.cos(2.0 * x2 * np.sin(np.pi * x1)) + x2 * x3 * x4
    )
    return (y0,)


def F5(x1, x2, x3, x4):
    """Equation for cart pendulum. Requires 4 hidden layers."""
    y1 = x3
    y2 = x4
    y3 = (
        -x1
        - 0.01 * x3
        + x4 ** 2 * np.sin(x2)
        + 0.1 * x4 * np.cos(x2)
        + 9.81 * np.sin(x2) * np.cos(x2)
    ) / (np.sin(x2) ** 2 + 1)
    y4 = (
        -0.2 * x4
        - 19.62 * np.sin(x2)
        + x1 * np.cos(x2)
        + 0.01 * x3 * np.cos(x2)
        - x4 ** 2 * np.sin(x2) * np.cos(x2) / (np.sin(x2) ** 2 + 1)
    )
    return (
        y1,
        y2,
        y3,
        y4,
    )


def save_fun(n=1, N=100):
    funs = [F1, F2, F3, F4, F5]
    x_train = torch.rand(N, 4) * 2 - 1
    y_train = funs[n - 1](x_train)
    data = (x_train, y_train)
    pickle.dump(data, gzip.open("data/f" + str(n) + "_10k.dat.gz", "wb"))


def gen_data(N=10_000):
    save_fun(1, N)
    save_fun(2, N)
    save_fun(3, N)


if __name__ == "__main__":
    gen_data()
