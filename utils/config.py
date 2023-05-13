from omegaconf import OmegaConf
from pathlib import Path

cfg = OmegaConf.create()
# 设置 ROOT_DIR 键的值
cfg.ROOT_DIR = str((Path(__file__).resolve().parent / '../').resolve())
