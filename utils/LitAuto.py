import warnings

from torch import nn

warnings.filterwarnings("ignore")
from typing import Any
import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryMatthewsCorrCoef,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix
)
import lightning.pytorch as pl
from utils.loss import FocalLoss
import json



# 从json文件中读取参数
with open('config.json', 'r') as f:
    config = json.load(f)

class LitAuto(pl.LightningModule):
    def __init__(self, module, model_params, lr=1e-4, weight_decay=0, **args) -> None:
        super().__init__()
        self.module = module(**model_params) if isinstance(module, type) else module
        self.save_hyperparameters(ignore="module")
        self.train_acc = BinaryAccuracy()
        self.train_mcc = BinaryMatthewsCorrCoef()
        # -----------------------------
        self.valid_acc = BinaryAccuracy()
        self.valid_auc = BinaryAUROC()
        self.valid_mcc = BinaryMatthewsCorrCoef()
        self.valid_f1 = BinaryF1Score()
        self.valid_recall = BinaryRecall()
        self.valid_prec = BinaryPrecision()
        self.valid_matrix = BinaryConfusionMatrix()

        # -----------------------------
        self.test_acc = BinaryAccuracy()
        self.test_auc = BinaryAUROC()
        self.test_mcc = BinaryMatthewsCorrCoef()
        self.test_f1 = BinaryF1Score()
        self.test_recall = BinaryRecall()
        self.test_prec = BinaryPrecision()
        self.test_matrix = BinaryConfusionMatrix()

    def forward(self, X):
        return self.module(X)

    def predict_step(self, batch, batch_idx) -> Any:
        return torch.sigmoid(self(batch[0]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        # loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        criterion = FocalLoss(alpha=config['alpha'], gamma=config["gamma"])
        loss=criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.train_mcc(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        # loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        criterion = FocalLoss(alpha=config['alpha'], gamma=config["gamma"])
        loss=criterion(y_hat, y)

        self.valid_acc(y_hat, y)
        self.valid_auc(y_hat, y)
        self.valid_mcc(y_hat, y)
        self.valid_f1(y_hat, y)
        self.valid_prec(y_hat, y)
        self.valid_recall(y_hat, y)
        self.valid_matrix(y_hat, y)
        self.log("valid_loss", loss)

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc)
        self.log("train_mcc", self.train_mcc)

    def on_validation_epoch_end(self) -> None:
        v_TP, v_FN, v_FP, v_TN = self.valid_matrix.compute()[[1, 1, 0, 0], [1, 0, 1, 0]]
        # 计算敏感性Sn
        v_Sn = v_TP / (v_TP + v_FN + 1e-06)
        # 计算特异性Sp
        v_Sp = v_TN / (v_FP + v_TN + 1e-06)

        auc = self.valid_auc.compute()
        mcc = self.valid_mcc.compute()
        prec = self.valid_prec.compute()
        recall = self.valid_recall.compute()
        PR =prec+recall
        S = 0.5 * auc + 0.5 * mcc
        NP=v_Sn+v_Sp

        self.log("valid_acc", self.valid_acc)
        self.log("valid_auc", self.valid_auc)
        self.log("valid_mcc", self.valid_mcc,prog_bar=True)
        self.log("valid_F1", self.valid_f1)
        self.log("valid_prec", self.valid_prec,prog_bar=True)
        self.log("valid_recall", self.valid_recall,prog_bar=True)
        self.log("valid_S", S)
        self.log("valid_Sn", v_Sn,prog_bar=True)
        self.log("valid_Sp", v_Sp,prog_bar=True)
        self.log("valid_NP",NP,prog_bar=True)
        self.log("valid_PR",PR,prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)

        self.test_acc(y_hat, y)
        self.test_auc(y_hat, y)
        self.test_mcc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_prec(y_hat, y)
        self.test_matrix(y_hat, y)

    def on_test_epoch_end(self) -> None:
        TP, FN, FP, TN = self.test_matrix.compute()[[1, 1, 0, 0], [1, 0, 1, 0]]
        # 计算敏感性Sn
        Sn = TP / (TP + FN + 1e-06)
        # 计算特异性Sp
        Sp = TN / (FP + TN + 1e-06)
        self.log("Sn", Sn)
        self.log("Sp", Sp)
        self.log("Acc", self.test_acc)
        self.log("AUC", self.test_auc)
        self.log("MCC", self.test_mcc)
        self.log("F1", self.test_f1)
        self.log("Recall", self.test_recall)
        self.log("Precision", self.test_prec)

    def configure_optimizers(self):
        if config['optimizer']== "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay  # type: ignore
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", factor=0.6, patience=7, min_lr=1e-5
        )
        configs = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": config["monitor"], "frequency": 1},
        }
        return configs
