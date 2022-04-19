import math
import torch
import torch.nn.functional as F
from torch import nn


class L0Dense(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        droprate=0.5,
        lamba=1.0,
        weight_decay=0.0,
        bias: bool = True,
        local_rep: bool = False,
    ):
        super().__init__()
        self.epsilon = 1e-6
        self.limit_a = -0.1
        self.limit_b = 1.1

        self.in_features = in_features
        self.out_features = out_features

        self.prior_prec = weight_decay
        self.local_rep = local_rep

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.ones(out_features))
        else:
            self.register_parameter("bias", None)

        self.qz_loga = nn.Parameter(torch.empty(out_features, in_features))

        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )

        self.droprate = droprate
        self.lamba = lamba
        self.temperature = 2.0 / 3.0
        self.reset_parameters()

    def reset_parameters(self):
        """Use pytorch linear for initialization (same member variable naming)"""
        torch.nn.Linear.reset_parameters(self)
        self.qz_loga.data.normal_(
            math.log(1 - self.droprate) - math.log(self.droprate), 1e-2
        )

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=self.epsilon, max=1 - self.epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        # logpw_col = torch.sum(- (.5 * self.prior_prec * self.weight.pow(2)) - self.lamba, 1)
        # logpw = torch.sum((1 - self.cdf_qz(0)).t() * logpw_col)
        # logpb = 0 if self.bias is None else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        # return logpw + logpb
        return self.lamba * torch.sum(1 - self.cdf_qz(0))

    def regularization(self):
        return self._reg_w()

    def regularization_constrained(self, idx):
        """Don't count L0 for every output but ignore reg for indices given by `idx`"""
        # set given cdf for given indices to 1 which then cancels the sum
        # could be implemented differently but this method doesn't require slicing/copying the cdf tensor
        cdf = self.cdf_qz(0)
        cdf[idx] = 1.0
        return self.lamba * (torch.sum(1 - cdf))

    def regularization_cheaper(self, idx, factor=0.5):
        """Make the L0 reg cheaper for indices given by `idx`"""
        cdf = self.cdf_qz(0)
        # cdf[idx] = torch.clamp_max(cdf[idx]*factor, 1.0)

        # decrase penalty for div units by factor
        reg_div = self.lamba * factor * torch.sum(1 - cdf[idx])
        non_div = cdf
        # div units won't result in penalty because 1-1 = 0
        non_div[idx] = 1.0
        reg_non_div = self.lamba * torch.sum(1 - non_div)

        # return self.lamba * (torch.sum(1 - cdf))
        return reg_div + reg_non_div

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # eps = torch.zeros_like(size).uniform_(self.epsilon, 1-self.epsilon)
        eps = self.floatTensor(size).uniform_(self.epsilon, 1 - self.epsilon)
        eps.requires_grad_()
        return eps

    def sample_z(self, batch_size: int = 1, sample: bool = True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            # eps = self.get_eps(torch.ones(batch_size, self.out_features, self.in_features))
            eps = self.get_eps(
                self.floatTensor(batch_size, self.out_features, self.in_features)
            )
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0.0, max_val=1.0)
        else:
            # deterministic, hence batch_size=1 is enough, i.e. batch_size param can be None
            # -> can't be none because of jit
            pi = torch.sigmoid(self.qz_loga).expand(self.out_features, self.in_features)
            return F.hardtanh(
                pi * (self.limit_b - self.limit_a) + self.limit_a,
                min_val=0.0,
                max_val=1.0,
            )

    def sample_weight(self):
        # z = self.quantile_concrete(self.get_eps(torch.ones(self.out_features, self.in_features)))
        z = self.quantile_concrete(
            self.get_eps(self.floatTensor(self.out_features, self.in_features))
        )
        mask = F.hardtanh(z, min_val=0.0, max_val=1.0)
        return mask * self.weight

    def deterministic_weight(self):
        # get deterministic z
        z = self.sample_z(sample=False)
        return self.weight * z

    def deterministic_bias(self):
        return self.bias

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(sample=self.training)
            output = F.linear(input, self.weight * z, self.bias)
        else:
            weight = self.sample_weight()
            output = F.linear(input, weight, self.bias)
        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class L0DenseBias(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        droprate=0.5,
        lamba=1.0,
        weight_decay=0.0,
        bias: bool = True,
        local_rep: bool = False,
        temperature=2.0 / 3.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.epsilon = 1e-6
        self.limit_a = -0.1
        self.limit_b = 1.1

        self.prior_prec = weight_decay
        self.local_rep = local_rep

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.qz_loga = nn.Parameter(torch.zeros(out_features, in_features + 1))

        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )

        self.droprate = droprate
        self.lamba = lamba
        self.temperature = temperature
        self.reset_parameters()

    def reset_parameters(self):
        """Use pytorch linear for initialization (same member variable naming)"""
        torch.nn.init.kaiming_uniform_(self.weight, mode="fan_out")
        self.qz_loga.data.normal_(
            math.log(1 - self.droprate) - math.log(self.droprate), 1e-2
        )

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=self.epsilon, max=1 - self.epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        # logpw_col = torch.sum(- (.5 * self.prior_prec * self.weight.pow(2)) - self.lamba, 1)
        # logpw = torch.sum((1 - self.cdf_qz(0)).t() * logpw_col)
        # logpb = 0 if self.bias is None else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        # return logpw + logpb
        return self.lamba * torch.sum(1 - self.cdf_qz(0))

    def regularization(self):
        return self._reg_w()

    def regularization_constrained(self, idx):
        """Don't count L0 for every output but ignore reg for indices given by `idx`"""
        # set given cdf for given indices to 1 which then cancels the sum
        # could be implemented differently but this method doesn't require slicing/copying the cdf tensor
        cdf = self.cdf_qz(0)
        cdf[idx] = 1.0
        return self.lamba * (torch.sum(1 - cdf))

    def regularization_cheaper(self, idx, factor=0.5):
        """Make the L0 reg cheaper for indices given by `idx`"""
        cdf = self.cdf_qz(0)
        # cdf[idx] = torch.clamp_max(cdf[idx]*factor, 1.0)

        # decrease penalty for div units by factor
        reg_div = self.lamba * factor * torch.sum(1 - cdf[idx])
        non_div = cdf
        # div units won't result in penalty because 1-1 = 0
        non_div[idx] = 1.0
        reg_non_div = self.lamba * torch.sum(1 - non_div)

        # return self.lamba * (torch.sum(1 - cdf))
        return reg_div + reg_non_div

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # eps = torch.zeros_like(size).uniform_(self.epsilon, 1-self.epsilon)
        eps = self.floatTensor(size).uniform_(self.epsilon, 1 - self.epsilon)
        eps.requires_grad_()
        # eps = Variable(eps)
        return eps

    def sample_z(self, batch_size: int = 1, sample: bool = True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            # eps = self.get_eps(torch.empty(batch_size, self.out_features, self.in_features+1))
            eps = self.get_eps(
                self.floatTensor(batch_size, self.out_features, self.in_features + 1)
            )
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0.0, max_val=1.0)
        else:  # deterministic, hence batch_size=1 is enough, i.e. batch_size param can be None
            pi = torch.sigmoid(self.qz_loga).expand(
                self.out_features, self.in_features + 1
            )
            return F.hardtanh(
                pi * (self.limit_b - self.limit_a) + self.limit_a,
                min_val=0.0,
                max_val=1.0,
            )

    def sample_weight(self):
        # z = self.quantile_concrete(self.get_eps(torch.empty(self.out_features, self.in_features+1)))
        z = self.quantile_concrete(
            self.get_eps(self.floatTensor(self.out_features, self.in_features + 1))
        )
        mask = F.hardtanh(z, min_val=0.0, max_val=1.0)[..., :, :-1]
        return mask * self.weight

    def sample_weight_bias(self):
        # z = self.quantile_concrete(self.get_eps(torch.empty(self.out_features, self.in_features+1)))
        z = self.quantile_concrete(
            self.get_eps(self.floatTensor(self.out_features, self.in_features + 1))
        )
        mask = F.hardtanh(z, min_val=0.0, max_val=1.0)
        return (
            mask[..., :, :-1] * self.weight,
            self.bias * mask[..., :, -1] if self.bias is not None else None,
        )

    def deterministic_weight(self):
        # get deterministic z
        z = self.sample_z(sample=False)
        return self.weight * z[:, :-1]

    def deterministic_bias(self):
        z = self.sample_z(sample=False)
        return self.bias * z[:, -1]

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(sample=self.training)
            output = F.linear(input, self.weight * z[:, :-1], self.bias * z[:, -1])
        else:
            weight, bias = self.sample_weight_bias()
            output = F.linear(input, weight, bias)
        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class L0DenseBiasCorr(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        droprate=0.5,
        lamba=1.0,
        weight_decay=0.0,
        bias: bool = True,
        local_rep: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.epsilon = 1e-6
        self.limit_a = -0.1
        self.limit_b = 1.1

        self.prior_prec = weight_decay
        self.local_rep = local_rep

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.qz_loga = nn.Parameter(torch.zeros(out_features, in_features + 1))

        ##self.cov = nn.Parameter(torch.ones(torch.numel(self.qz_loga)))

        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )

        self.droprate = droprate
        self.lamba = lamba
        self.temperature = 2.0 / 3.0
        self.reset_parameters()

    def reset_parameters(self):
        """Use pytorch linear for initialization (same member variable naming)"""
        torch.nn.init.kaiming_uniform_(self.weight, mode="fan_out")
        self.qz_loga.data.normal_(
            math.log(1 - self.droprate) - math.log(self.droprate), 1e-2
        )

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=self.epsilon, max=1 - self.epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        # logpw_col = torch.sum(- (.5 * self.prior_prec * self.weight.pow(2)) - self.lamba, 1)
        # logpw = torch.sum((1 - self.cdf_qz(0)).t() * logpw_col)
        # logpb = 0 if self.bias is None else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        # return logpw + logpb
        return self.lamba * torch.sum(1 - self.cdf_qz(0))

    def regularization(self):
        return self._reg_w()

    def regularization_constrained(self, idx):
        """Don't count L0 for every output but ignore reg for indices given by `idx`"""
        # set given cdf for given indices to 1 which then cancels the sum
        # could be implemented differently but this method doesn't require slicing/copying the cdf tensor
        cdf = self.cdf_qz(0)
        cdf[idx] = 1.0
        return self.lamba * (torch.sum(1 - cdf))

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # eps = torch.zeros_like(size).uniform_(self.epsilon, 1-self.epsilon)
        eps = self.floatTensor(size).uniform_(self.epsilon, 1 - self.epsilon)
        eps.requires_grad_()
        # eps = Variable(eps)
        return eps

    def sample_z(self, batch_size: int = 1, sample: bool = True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            # eps = self.get_eps(torch.empty(batch_size, self.out_features, self.in_features+1))
            eps = self.get_eps(
                self.floatTensor(batch_size, self.out_features, self.in_features + 1)
            )
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0.0, max_val=1.0)
        else:  # deterministic, hence batch_size=1 is enough, i.e. batch_size param can be None
            pi = torch.sigmoid(self.qz_loga).expand(
                self.out_features, self.in_features + 1
            )
            return F.hardtanh(
                pi * (self.limit_b - self.limit_a) + self.limit_a,
                min_val=0.0,
                max_val=1.0,
            )

    def sample_weight(self):
        # z = self.quantile_concrete(self.get_eps(torch.empty(self.out_features, self.in_features+1)))
        z = self.quantile_concrete(
            self.get_eps(self.floatTensor(self.out_features, self.in_features + 1))
        )
        mask = F.hardtanh(z, min_val=0.0, max_val=1.0)[..., :, :-1]
        return mask * self.weight

    def sample_weight_bias(self):
        # z = self.quantile_concrete(self.get_eps(torch.empty(self.out_features, self.in_features+1)))
        z = self.quantile_concrete(
            self.get_eps(self.floatTensor(self.out_features, self.in_features + 1))
        )
        mask = F.hardtanh(z, min_val=0.0, max_val=1.0)
        return (
            mask[..., :, :-1] * self.weight,
            self.bias * mask[..., :, -1] if self.bias is not None else None,
        )

    def deterministic_weight(self):
        # get deterministic z
        z = self.sample_z(sample=False)
        return self.weight * z[:, :-1]

    def deterministic_bias(self):
        z = self.sample_z(sample=False)
        return self.bias * z[:, -1]

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(sample=self.training)
            output = F.linear(input, self.weight * z[:, :-1], self.bias * z[:, -1])
        else:
            weight, bias = self.sample_weight_bias()
            output = F.linear(input, weight, bias)
        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
