import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from opt import get_args
from utils.config import config, cfg_from_yaml_file, cfg_update_from_cli
from utils.training_utils import set_random_seed

import datetime
from utils.pytorch_lighting_utils import ImageLogCallback
from utils.utils import make_source_code_snapshot

from dataset import build_lightning_data_module


def train(config):
    # system = build_system(config)
    # data = build_data(config.data)
    data_module = build_lightning_data_module(config.data)

    print(f"Start with exp_name: {config.name}.")
    print(os.path.join(config.log_dir, config.group))
    logger = TensorBoardLogger(save_dir=os.path.join(config.log_dir, config.group), name=config.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.exp_path,
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    callbacks = [LearningRateMonitor("step"), ImageLogCallback(), checkpoint_callback]
    make_source_code_snapshot(f"{config.exp_path}")
    OmegaConf.save(config=config, f=os.path.join(config.exp_path, "run_config_snapshot.yaml"))

    # trainer = Trainer(
    #     max_epochs=config.train.num_epochs,  # 最大训练轮数
    #     callbacks=callbacks,  # 回调函数
    #     resume_from_checkpoint=config.ckpt_path,  # 从checkpoint恢复
    #     logger=logger,  # 日志
    #     enable_model_summary=False,  # 是否打印模型结构
    #     gpus=config.train.num_gpus,  # 使用的GPU数量
    #     accelerator="ddp" if config.train.num_gpus > 1 else None,  # 使用的加速器
    #     num_sanity_val_steps=1,  # 验证集的batch数
    #     benchmark=True,  # 是否开启benchmark
    #     profiler="simple" if config.train.num_gpus == 1 else None,  # 是否开启profiler
    #     val_check_interval=1,  # 验证集的检查间隔
    #     log_every_n_steps=50,  # logger的间隔
    #     precision=config.precision,  # 半精度加速
    # )

    # trainer.fit(system, data)


if __name__ == '__main__':
    args = get_args()  # 获取命令行参数

    # 从yaml文件中加载配置
    cfg_from_yaml_file(config=config, cfg_file=args.cfg_file)
    cfg_update_from_cli(config=config, args=args)
    config.merge_with_dotlist(args.opts)
    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed)

    config.name = config.name + datetime.datetime.now().strftime("%mM_%dD_%HH_%MM") + "_seed" + str(config.seed)
    # 设置日志路径

    config.exp_path = os.path.join(config.log_dir, config.group, config.name)
    print(config.exp_path)
    train(config)
