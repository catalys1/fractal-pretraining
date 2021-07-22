from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
from torchmetrics import Accuracy, F1

from .base import ClassificationModel


class MultiLabelModel(ClassificationModel):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        warmup: float = 0.1,
        training_steps: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        optim_name: str = 'SGD',
        optim_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        loss_fn = loss_fn if loss_fn is not None else torch.nn.functional.binary_cross_entropy_with_logits
        super().__init__(
            model, lr, weight_decay, warmup, training_steps, loss_fn, optim_name, optim_kwargs
        )

    def setup_metrics(self):
        self.train_accuracy = Accuracy()
        self.train_f1 = F1()
        self.val_accuracy = Accuracy()
        self.val_f1 = F1()
        self.metric_hist = {
            "train/acc": [],
            "train/loss": [],
            "train/f1": [],
            "val/acc": [],
            "val/loss": [],
            "val/f1": [],
        }

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        y = torch.zeros(logits.shape, dtype=x.dtype, device=x.device).scatter_(1, y, 1)
        w = torch.empty(logits.shape[1], dtype=x.dtype, device=x.device).fill_(200)
        loss = self.loss_fn(logits, y, pos_weight=w)
        with torch.no_grad():
            preds = logits.sigmoid()
        return loss, preds, y.int()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/f1", f1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/f1"].append(self.trainer.callback_metrics["train/f1"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/f1_best", max(self.metric_hist["train/f1"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        f1 = self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", f1, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/f1"].append(self.trainer.callback_metrics["val/f1"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/f1_best", max(self.metric_hist["val/f1"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)
