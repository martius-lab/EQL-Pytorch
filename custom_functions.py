import torch
import sympy as sy

"""
This file contains custom functions that are not directly available
in torch/sympy. Most functions need a separate torch and a sympy version, 
except for example polynomials.
"""


def func_cube(x):
    return x * x * x


def func_square(x):
    return x * x


def func_id(x):
    return x


def log_reg(x):
    return torch.log(torch.clamp(x + 1.0, min=0.001))


def log_reg_sy(x):
    return sy.log(x + 1)


def exp_reg(x):
    return torch.exp(torch.clamp(x, max=10.0)) - torch.ones_like(x)


def sqrt_reg(x):
    return torch.sqrt(torch.abs(x) + 1e-8)


def sqrt_reg_sy(x):
    return sy.sqrt(sy.Abs(x))


def exp_reg_sy(x):
    return sy.exp(x) - 1


def sing_div(x):
    mask = torch.abs(x) > 1e-2
    return (
        torch.nan_to_num(torch.div(1.0, torch.abs(x) + 1e-6), posinf=1e5, neginf=-1e5)
        * mask
    )


def sing_div_sy(x):
    return 1 / x


def div_reg(x, y):
    return y.sign() * (torch.div(x, torch.clip_(y.abs(), min=1e-4)))


def div_reg_sy(x, y):
    return sy.Symbol.__truediv__(x, y)


def v_add(x, y):
    return x + y


def v_sub(x, y):
    return x - y


def v_square(x, y):
    return x ** 2 + y ** 2
