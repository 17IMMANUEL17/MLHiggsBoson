""" Main logic to run the project """

import argparse
import os

from core.data_preparation import prepare_data
from core.trainer import run_training
from tools.cfg_parser import Config
from tools.utils import makelogger, makepath


def run_higgs(cfg):
    """Runs main logic of the project"""
    logger = makelogger(
        makepath(os.path.join(cfg.work_dir, f"{cfg.expr_ID}.log"), isfile=True)
    )
    logger.info(f"[{cfg.expr_ID}] - ML-Higgs experiment has started!")
    train_data, test_data = prepare_data(cfg)

    # run the training and evaluation of the models
    run_training(cfg, train_data, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train-ML-Higgs")
    parser.add_argument(
        "--data-path", required=True, type=str, help="Path to the folder with data"
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=str,
        help="Path to the working directory where the results will be saved",
    )
    parser.add_argument("--expr-ID", default="V00", type=str, help="Training ID")

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, "configs/training_cfg.yaml")

    args = parser.parse_args()
    config = {
        "dataset_dir": args.data_path,
        "expr_ID": args.expr_ID,
        "work_dir": os.path.join(args.work_dir, args.expr_ID),
        "model_selection": {
            "mean_squared_error_gd": False,
            "mean_squared_error_sgd": False,
            "least_squares": False,
            "ridge_regression": False,
            "logistic_regression": False,
            "reg_logistic_regression": False,
            "logistic_regression_bfgs": False,
            "reg_logistic_regression_bfgs": True,
        },
        "n_epochs": 400,
        "grid_search": False,
        "cross_validation": True,
        "polynomial_features": True,
    }

    config = Config(default_cfg_path=default_cfg_path, **config)
    config.write_cfg(os.path.join(config.work_dir, "config.yaml"))
    run_higgs(config)
