from turtle import ycor
import torch
import torch.nn
import sympy as sy
from typing import List, Tuple, Callable
from torch.nn.functional import softplus
import numpy as np
import custom_functions
from l0_dense import L0DenseBias, L0Dense


# torch to sympy
f_dict_sy = {
    torch.sin: sy.sin,
    torch.cos: sy.cos,
    custom_functions.func_id: sy.Id,
    torch.mul: sy.Symbol.__mul__,
    torch.div: sy.Symbol.__truediv__,
    torch.log: sy.log,
    torch.exp: sy.exp,
    torch.sqrt: sy.sqrt,
    custom_functions.sqrt_reg: custom_functions.sqrt_reg_sy,
    custom_functions.log_reg: custom_functions.log_reg_sy,
    custom_functions.exp_reg: custom_functions.exp_reg_sy,
    custom_functions.div_reg: custom_functions.div_reg_sy,
    custom_functions.sing_div: custom_functions.sing_div_sy,
    custom_functions.func_square: custom_functions.func_square,
    custom_functions.func_cube: custom_functions.func_cube,
    custom_functions.v_add: custom_functions.v_add,
    custom_functions.v_sub: custom_functions.v_sub,
    custom_functions.v_square: custom_functions.v_square,
}


# string to torch + arity
f_dict_torch = {
    "sin": (torch.sin, 1),
    "cos": (torch.cos, 1),
    "id": (custom_functions.func_id, 1),
    "mul": (torch.mul, 2),
    "div": (torch.div, 2),
    "log": (torch.log, 1),
    "exp": (torch.exp, 1),
    "sqrt": (torch.sqrt, 1),
    "log_reg": (custom_functions.log_reg, 1),
    "exp_reg": (custom_functions.exp_reg, 1),
    "sing_div": (custom_functions.sing_div, 1),
    "div_reg": (custom_functions.div_reg, 2),
    "sqrt_reg": (custom_functions.sqrt_reg, 1),
    "square": (custom_functions.func_square, 1),
    "cube": (custom_functions.func_cube, 1),
}


def get_indices(
    functions: List[Tuple[Callable, int]]
) -> Tuple[List[int], List[List[int]]]:
    """
    Gets a list of function and then assigns to each function
    an index to act on. Returns indices for unary and binary
    separately.
    """
    unary_indices = []
    binary_indices = []
    i = 0

    # augment index by arity of operator
    for _, a in functions:
        if a == 1:
            unary_indices.append(i)
            i += 1
        elif a == 2:
            binary_indices.append([i, i + 1])
            i += 2
        else:
            raise KeyError
    return unary_indices, binary_indices


def round_floats(expr, to: int = 3):
    """
    Takes a sympy expression and rounds every float to
    `to` digits
    """
    new_expr = expr
    # walks along the expression tree
    for a in sy.preorder_traversal(expr):
        if isinstance(a, sy.Float):
            new_expr = new_expr.subs(a, round(a, to))
    return new_expr


def pretty_function(model):
    expr = model.get_symbolic_expr()[0]
    return round_floats(sy.expand(expr))


class EQLLayerBase(torch.nn.Module):
    """
    Base layer for the equation learner. Implements the forward method and
    the symbolic expression creation. _Cannot_ be used without subclassing and specifying the type
    of linear layer.
    """

    def __init__(self, in_features: int, functions: List[str]):
        """
        :param in_features: Number of input features
        :param functions: List of functions that are used at nodes
        """
        super().__init__()
        self.in_features = in_features

        # convert function string to function through dict
        functions = [f_dict_torch[f] for f in functions]

        # sum arity of all functions for num of params
        self.number_of_vars = sum(a for f, a in functions)
        self.functions = functions
        self.number_of_functions = len(functions)

        # indices of cols that respective functions act on.
        # split unary and binary
        self.unary_indices, self.binary_indices = get_indices(functions)

        # override this member to use
        self.linear_layer = None

        # combine function and respective index into tuple
        # type annotations to try to help the torch.jit :/
        # List[Tuple[Callable, int]]
        self.unary_funcs = [
            (func, index)
            for func, index in zip(
                (f for f, a in functions if a == 1), self.unary_indices
            )
        ]
        self.binary_funcs = [
            (func, index)
            for func, index in zip(
                (f for f, a in functions if a == 2), self.binary_indices
            )
        ]

        self.num_unary_funcs = len(self.unary_funcs)
        self.num_binary_funcs = len(self.binary_funcs)

        self.threshold = 1e-2

    def forward(self, x):
        z = self.linear_layer(x)

        # operate with each function on one column. binary functions need two columns.
        # at least one unary function is expected. no binary op is also ok
        # x can have arbitrary dimension, just work on the last one

        unary_stack = torch.stack([f(z[..., i]) for f, i in self.unary_funcs], -1)
        if self.binary_funcs:
            binary_stack = torch.stack(
                [f(z[..., i[0]], z[..., i[1]]) for f, i in self.binary_funcs], -1
            )

            # concatenate to resulting matrix
            y = torch.cat((unary_stack, binary_stack), -1)

        else:
            y = unary_stack
        return y

    def get_weight_bias(self):
        """
        Return the weight and bias of the (only) linear map in the
        layer
        """
        raise NotImplementedError

    def get_symbolic_expr(self, var_name="x"):
        """
        Constructs a sympy representation of the function described
        by the layer
        """
        with torch.no_grad():
            w, b = self.get_weight_bias()

            in_symbols = sy.symbols("{}:{}".format(var_name, self.in_features))
            z = []
            for i in range(self.number_of_vars):
                o = 0
                for j in range(self.in_features):
                    o += in_symbols[j] * w[i, j].item()
                if b is not None:
                    o += b[i].item()
                z.append(o)

            outs = []
            for f, i in self.unary_funcs:
                s = f_dict_sy[f](z[i])
                # if s == sy.zoo:
                #    s = 0
                outs.append(s)

            for f, i in self.binary_funcs:
                s = f_dict_sy[f](z[i[0]], z[i[1]])
                outs.append(s)

        return outs


class EQLLayerDefault(EQLLayerBase):
    """
    Uses standard nn.Linear for the linear layer
    """

    def __init__(self, in_features, functions, bias=True):
        super().__init__(in_features, functions)
        self.linear_layer = torch.nn.Linear(in_features, self.number_of_vars, bias=bias)
        self.bias = bias
        self.reset_parameters()

    def get_weight_bias(self):
        return (self.linear_layer.weight, self.linear_layer.bias if self.bias else None)

    def reset_parameters(self):
        # use normal distribution to initialize weights
        torch.nn.init.normal_(self.linear_layer.weight, mean=0.0, std=1.0)
        if self.bias:
            torch.nn.init.uniform_(self.linear_layer.bias)

    def get_active_params(self):
        """
        Returns the number of active weights in the weight matrix
        """
        ll = self.linear_layer
        nonzero = ll.weight.detach().count_nonzero().item()
        nonzero += ll.bias.detach().count_nonzero().item()
        total = (ll.in_features + 1) * ll.out_features
        return nonzero, total

    def get_l1_reg(self):
        return torch.norm(self.linear_layer.weight, 1)


class EQLLayerL0(EQLLayerBase):
    """
    Uses a L0 dense layer for the linear layer
    """

    def __init__(self, in_features, functions, bias=True, droprate=0.5, **kwargs):
        super().__init__(in_features, functions)
        self.linear_layer = L0DenseBias(
            in_features,
            self.number_of_vars,
            weight_decay=0.0,
            droprate=droprate,
            bias=bias,
            **kwargs
        )
        self.bias = bias

    def get_weight_bias(self):
        return (
            self.linear_layer.deterministic_weight(),
            self.linear_layer.deterministic_bias() if self.bias else None,
        )

    def get_l0_reg(self):
        return self.linear_layer.regularization()

    def get_active_params(self):
        """
        Returns the number of active weights in the weight matrix
        """
        ll = self.linear_layer
        z = ll.sample_z(1, sample=False).detach()
        nonzero = z.count_nonzero().item()
        total = (ll.in_features + 1) * ll.out_features
        return nonzero, total


class EQL(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        functions: List[List[str]],
        n_layers=1,
        use_l0=False,
        bias=True,
        **kwargs
    ):
        """
        Constructs an EQL network of arbitrary layer size.  Every layer can have different functions.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param functions: (List) of list of functions used at the nodes
        :param n_layers: Number of layers
        :param use_l0: Whether to use a L0 layer for the linear layers
        :param bias: Use bias within the linear layers
        """
        super().__init__()
        base = EQLLayerDefault if not use_l0 else EQLLayerL0

        self.in_features = in_features
        self.out_features = out_features
        self.use_l0 = use_l0

        # use the same function for all layers
        if not isinstance(functions[0], List):
            functions = [functions] * n_layers

        # first layer
        layers = torch.nn.ModuleList(
            [base(in_features, functions[0], bias=bias, **kwargs)]
        )

        # inner eql layers
        for i in range(1, n_layers):
            layers.append(
                base(layers[-1].number_of_functions, functions[i], bias=bias, **kwargs)
            )

        """
        add last layer to create output. can either be normal layer without pruning or a 
        L0 layer
        """
        if self.use_l0:
            layers.append(
                L0DenseBias(
                    layers[-1].number_of_functions, out_features, bias=bias, **kwargs
                )
            )
        else:
            layers.append(
                torch.nn.Linear(layers[-1].number_of_functions, out_features, bias=bias)
            )

        self.layers = layers

        # mask in case they are used
        self.masks = [None] * len(self.layers)

    def forward(self, input):
        for module in self.layers:
            input = torch.nan_to_num(module(input), nan=0.0, posinf=1e5, neginf=-1e5)
        return input

    def get_symbolic_expr(self):
        with torch.no_grad():
            a = self.layers[0].get_symbolic_expr("x")
            for module in self.layers[1:-1]:
                b = module.get_symbolic_expr("b")
                c = []
                for i in range(len(b)):
                    c.append(b[i].subs({"b" + str(j): a[j] for j in range(len(a))}))
                a = c

            # get weight/bias of last (normal) linear layer
            w, b = self.layers[-1].weight, self.layers[-1].bias
            z = []

            for i in range(self.out_features):
                s = 0
                for j in range(self.layers[-2].number_of_functions):
                    s += a[j] * w[i, j].item()
                if b is not None:
                    s += b[i].item()
                z.append(s)
        return z

    def set_masks(self, threshold):
        """
        Sets the masks for all layer weights matrices. Entries that are smaller than
        `threshold` are set to zero.
        """
        for i, l in enumerate(self.layers[:-1]):
            self.masks[i] = torch.abs(l.linear_layer.weight) < threshold

        # last layer is no EQLLayer (just linear) hence extra treatment
        self.masks[-1] = torch.abs(self.layers[-1].weight) < threshold

    def update_masks(self, threshold):
        """
        Updates masks, i.e. masked unit will remain masked and cannot become unmasked as in the `set_masks`
        method.
        """
        for i, l in enumerate(self.layers[:-1]):
            newmask = torch.abs(l.linear_layer.weight) < threshold
            self.masks[i] = torch.bitwise_or(self.masks[i], newmask)

        # last layer is no EQLLayer (just linear) hence extra treatment
        self.masks[-1] = torch.bitwise_or(
            self.masks[-1], (torch.abs(self.layers[-1].weight) < threshold)
        )

    def apply_masks(self):
        """
        All matrix elements/weights that are masked (True) are set to zero.
        """
        for i, l in enumerate(self.layers[:-1]):
            l.linear_layer.weight.data[self.masks[i]] = 0.0

        self.layers[-1].weight.data[self.masks[-1]] = 0.0

    def reset_masks(self):
        for m in self.masks:
            m.data = torch.ones_like(m) * True

    def get_l0_reg(self, cheaper_div=False, factor=0.25):
        if not self.use_l0:
            raise NotImplementedError

        reg = 0.0
        for module in self.layers[:-1]:
            reg += module.get_l0_reg()

        reg += self.layers[-1].regularization()

        return reg

    def get_l1_reg(self):
        if self.use_l0:
            raise NotImplementedError

        reg = 0.0
        for module in self.layers[:-1]:
            reg += module.get_l1_reg()

        return reg

    def set_to(self, val):
        # TODO: functions not defined for base layers as of now
        for l in self.layers[:-1]:
            l.set_to(val)

        if self.use_l0:
            self.layers[-1].set_to(val)
        else:
            torch.nn.init.constant_(self.layers[-1].weight, val)
            torch.nn.init.constant_(self.layers[-1].bias, 0.0)

    def get_active_params(self):
        # NOTE: last layer is not counted, not an EQL layer
        nonzero = total = 0
        for l in self.layers[:-1]:
            n, t = l.get_active_params()
            nonzero += n
            total += t
        return nonzero, total
