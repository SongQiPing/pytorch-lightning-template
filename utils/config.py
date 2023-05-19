from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type
from dataclasses import dataclass

config = OmegaConf.create()
# 设置 ROOT_DIR 键的值
config.ROOT_DIR = str((Path(__file__).resolve().parent / '../').resolve())


def merge_new_config(config, new_config):
    if '__BASE_CONFIG__' in new_config:
        # with open(new_config['_BASE_CONFIG_'], 'r') as f:
        base_config_path = new_config['__BASE_CONFIG__']
        base_config = OmegaConf.load(base_config_path)
        new_config.pop('__BASE_CONFIG__')
        new_config = OmegaConf.merge(base_config, new_config)
    config.merge_with(new_config)
    return config


def cfg_from_yaml_file(config, cfg_file):
    new_config = OmegaConf.load(cfg_file)

    merge_new_config(config=config, new_config=new_config)

    return config


def cfg_update_from_cli(config, args):
    config_cli = OmegaConf.create(vars(args))
    config.merge_with(config_cli)
    config.merge_with_dotlist(args.opts)
    return config


class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


if __name__ == '__main__':
    config = cfg_from_yaml_file('configs/MNIST.yaml', config)
    print(config)
