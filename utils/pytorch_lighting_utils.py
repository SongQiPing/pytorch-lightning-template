from pytorch_lightning import Callback
from pytorch_lightning import Trainer
import torch


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""
        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                # trainer.logger.experiment.log({"images": wandb.Image([images])}, commit=False)
                trainer.logger.experiment.add_image('images', images, trainer.global_step)

    def on_test_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""
        self.on_validation_epoch_end(trainer, pl_module)