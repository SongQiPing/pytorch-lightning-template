import argparse
import sys


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--ckpt_path", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    parser.add_argument("--is_logger_enabled", default=True, type=bool,
                        help="whether to enable logger")

    # 添加半精度
    parser.add_argument("--precision", default=32, type=int, help="precision")
    return parser.parse_args()
