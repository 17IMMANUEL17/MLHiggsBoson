""" Main logic to run the project """

import argparse
import os

from tools.cfg_parser import Config
from data_preparation import prepare_data


def run_higgs(cfg):
    """ Runs main logic of the project """
    print(cfg)
    prepare_data(cfg.dataset_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-ML-Higgs')
    parser.add_argument('--data-path', default='./data', required=False, type=str,
                        help='Path to the folder with data')

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, 'configs/training_cfg.yaml')

    args = parser.parse_args()
    config = {
        'dataset_dir': args.data_path,
    }

    config = Config(default_cfg_path=default_cfg_path, **config)
    run_higgs(config)
