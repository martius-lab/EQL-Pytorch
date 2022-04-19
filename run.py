# import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

import torch
from eqlearner import EQL
import pickle
import gzip
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
import sympy as sy
from types import SimpleNamespace as sn


class LitEQL(pl.LightningModule):
    """
    use masking to prune equations
    """

    def __init__(self, units, reg, mask_thresh, total_epochs):
        super().__init__()
        x_in = 4
        y_out = 1
        self.reg = reg
        self.model = EQL(x_in, y_out, units)

        self.total_epochs = total_epochs
        self.T1 = self.total_epochs // 4
        self.T2 = (self.total_epochs * 19) // 20

        self.mask_thresh = mask_thresh
        self.loss = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        if self.current_epoch == self.T2:
            self.model.set_masks(self.mask_thresh)

        mse_loss = F.mse_loss(y_pred, y)
        l1_loss = self.model.get_l1()
        loss = mse_loss

        if self.T1 < self.current_epoch < self.T2:
            loss += self.reg * l1_loss

        if self.current_epoch > self.T2:
            self.model.apply_masks()

        self.log("mse_loss", mse_loss)
        self.log("l1_loss", l1_loss)
        self.loss = loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LitEQL2(pl.LightningModule):
    """
    use l0 to prune equations
    """

    def __init__(self, units, reg, total_epochs):
        super().__init__()
        x_in = 4
        y_out = 1
        self.reg = reg
        self.model = EQL(x_in, y_out, units, use_l0=True)

        self.total_epochs = total_epochs
        self.loss = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        mse_loss = F.mse_loss(y_pred, y)
        l0_loss = self.model.get_l0_reg()
        loss = mse_loss + self.reg * l0_loss

        self.log("mse_loss", mse_loss)
        self.log("l0_loss", l0_loss)

        self.loss["mse"] = mse_loss.item()
        self.loss["l0"] = l0_loss.item()

        tensorboard = self.logger.experiment
        p, t = self.model.get_active_params()
        tensorboard.add_text("active params", str(p) + " of " + str(t))
        self.loss["parameters"] = p

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(working_dir, eql_params, train_params):
    data = pickle.load(gzip.open("data/f1_10k.dat.gz", "rb"))
    dataset = TensorDataset(*data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    eql = LitEQL2(
        **eql_params, reg=train_params.l0_reg, total_epochs=train_params.iterations
    )
    tb_logger = pl_loggers.TensorBoardLogger(working_dir + "/logs/")
    trainer = pl.Trainer(max_epochs=train_params.iterations, logger=tb_logger)
    trainer.fit(eql, train_loader)

    func = str(sy.expand(eql.model.get_symbolic_expr()[0]))
    print(func, file=open(working_dir + "/func.txt", "w"))
    print(eql.loss)
    return 0


if __name__ == "__main__":
    working_dir = "./"
    eql_params = {"units": ["id", "id", "sin", "cos", "mul", "cos", "sin"]}
    train_params = {"l0_reg": 1e-4, "iterations": 100}
    main(working_dir, eql_params, sn(**train_params))
