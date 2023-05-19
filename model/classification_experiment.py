import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import torch.optim.lr_scheduler as lrs
from torchmetrics import Accuracy

class ClassificationExperiment(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.save_hyperparameters(hparams)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)
        loss = F.nll_loss(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(y.view_as(pred)).sum() / (x.shape[0] * 1.0)
        return {"batch_test_acc": accuracy}

    def configure_optimizers(self):
        optimizer_hparams = self.hparams.optimizer
        if hasattr(optimizer_hparams, 'weight_decay'):
            weight_decay = optimizer_hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=optimizer_hparams.lr, weight_decay=weight_decay)

        if optimizer_hparams.lr_scheduler is None:
            return optimizer
        else:
            if optimizer_hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=optimizer_hparams.lr_decay_steps,
                                       gamma=optimizer_hparams.lr_decay_rate)
            elif optimizer_hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=optimizer_hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def test_epoch_end(self, outputs):
        accuracy = torch.stack([x['batch_test_acc'].float() for x in outputs]).mean()
        return {"log": {"test_acc": accuracy}}
