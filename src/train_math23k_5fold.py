from dataset import loadMath23K_5fold
from solver import MathSolverRNN, MathSolverRE
from trainer import MathTrainerRNN, MathTrainerRE
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

fold_Acc = []

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
    test_dataset: List[Dict],
    fold: int,
    cfg: MathConfig,
    solver: Union[MathSolverRNN, MathSolverRE],
):
    # trainer = MathTrainer(cfg, train_dataset, test_dataset)
    if args.model_type == "re":
        trainer = MathTrainerRE(cfg, train_dataset, test_dataset, use_dev=False)
    elif args.model_type == "rnn":
        trainer = MathTrainerRNN(cfg, train_dataset, test_dataset, use_dev=False)
    else:
        raise ValueError

    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-math23k_5fold-{}-{}-rev".format(fold, args.model_type))
    logger.info("[finish train solver]")
    logger.info("best test acc: {}".format(trainer.best_test_acc))
    fold_Acc.append(trainer.best_test_acc)


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    logger.info("model type: {}".format(args.model_type))
    
    for fold in reversed(range(5)):
        train_dataset, test_dataset, const_nums = loadMath23K_5fold(args.data_path, fold, head=args.head)
        
        logger.info("fold: {}".format(fold))
        logger.info("train dataset size: {}".format(len(train_dataset)))
        logger.info("test dataset size: {}".format(len(test_dataset)))
        
        cfg = MathConfig(**json.loads(args.cfg))
        cfg.dataset_name = args.dataset_name
        cfg.debug = args.debug
        cfg.const_quant_size = len(const_nums)
        cfg.ext_tokens = ['^']
        
        logger.info("len(const_quant_size): {}".format(len(const_nums)))
        logger.info("const_quants: {}".format(const_nums))

        # solver = MathSolver(cfg, const_nums)
        solver_dict = {
            "re": MathSolverRE,
            "rnn": MathSolverRNN,
        }
        if args.model_type in solver_dict:
            solver = solver_dict[args.model_type](cfg)
        else:
            raise ValueError(args.model_type)
                
        train_solver(args, train_dataset, test_dataset, fold, cfg, solver)

    fold_mAcc = sum(fold_Acc) / len(fold_Acc)
    logger.info("5-fold mAcc: {}".format(fold_mAcc))


if __name__ == '__main__':
    main(get_args())
