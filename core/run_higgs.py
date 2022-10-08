""" Main logic to run the project """

import argparse
import os

from tools.cfg_parser import Config
from data_preparation import prepare_data

from tools.utils import makelogger, makepath


def run_higgs(cfg):
    """ Runs main logic of the project """
    logger = makelogger(makepath(os.path.join(cfg.work_dir, f'{cfg.expr_ID}.log'), isfile=True))
    logger.info(f'[{cfg.expr_ID}] - ML-Higgs experiment has started!')
    prepare_data(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-ML-Higgs')
    parser.add_argument('--data-path', required=True, type=str,
                        help='Path to the folder with data')
    parser.add_argument('--work-dir', required=True, type=str,
                        help='Path to the working directory where the results will be saved')
    parser.add_argument('--expr-ID', default='V00', type=str,
                        help='Training ID')

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, 'configs/training_cfg.yaml')

    args = parser.parse_args()
    config = {
        'dataset_dir': args.data_path,
        'expr_ID': args.expr_ID,
        'work_dir': os.path.join(args.work_dir, args.expr_ID),
    }

    config = Config(default_cfg_path=default_cfg_path, **config)
    run_higgs(config)
