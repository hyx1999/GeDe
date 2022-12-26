from dataset import loadMathQA
from solver import MathSolverRPD, MathSolverRPE, MathSolverRPE_Abl0
from trainer import MathTrainerRPD, MathTrainerRPE, MathTrainerRPE_Abl0
from cfg import MathConfig

import datetime
import argparse
import json
from loguru import logger
from typing import Dict, List, Union
from copy import deepcopy
from tqdm import tqdm

import random
import torch
import numpy as np


def setup_logger():
    format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    format="[{time}]-[{level}] {file}-{function}-{line}: {message}"
    logger.remove(None)
    logger.add(
        f"../log/{format_time}.log",
        rotation="100 MB", 
        level="INFO", 
        format=format
    )


def setup_seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="test")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--log_text", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--load_model_dir", type=str, required=True)
    parser.add_argument("--save_model_dir", type=str, required=True)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--cfg", type=str, default="{}")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def train_solver(
    args: argparse.Namespace,
    train_dataset: List[Dict],
    dev_dataset: List[Dict],
    test_dataset: List[Dict],
    cfg: MathConfig,
    solver: Union[MathSolverRPD, MathSolverRPE],
):
    trainer_dict = {
        "rpe": MathTrainerRPE,
        "rpe_abl0": MathTrainerRPE_Abl0,
        "rpd": MathTrainerRPD,
    }
    if args.model_type in trainer_dict:
        trainer = trainer_dict[args.model_type](cfg, train_dataset, test_dataset, dev_dataset=dev_dataset)
    else:
        raise ValueError(args.model_type)

    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-mathqa-{}".format(args.model_type))
    logger.info("[finish train solver]")
    logger.info("best test acc: {}".format(trainer.best_test_acc))


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    logger.info("model type: {}".format(args.model_type))
    
    train_dataset, dev_dataset, test_dataset, const_nums = loadMathQA(args.data_path, head=args.head)
    
    cfg = MathConfig(**json.loads(args.cfg))
    cfg.dataset_name = args.dataset_name
    cfg.debug = args.debug
    cfg.const_quant_size = len(const_nums)
    cfg.ext_tokens = ['^']
    
    logger.info("len(const_quant_size): {}".format(len(const_nums)))
    logger.info("const_quants: {}".format(const_nums))

    solver_dict = {
        "rpe": MathSolverRPE,
        "rpe_abl0": MathSolverRPE_Abl0,
        "rpd": MathSolverRPD,
    }
    if args.model_type in solver_dict:
        solver = solver_dict[args.model_type](cfg)
    else:
        raise ValueError(args.model_type)
    
    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, dev_dataset, test_dataset, cfg, solver)


if __name__ == '__main__':
    main(get_args())
