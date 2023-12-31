from dataset import loadMath23KRaw, build_ext_words, join_const_nums, join_Expr_list
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
    parser.add_argument("--expr_mode", type=str, default="v3")
    return parser.parse_args()


def train_solver(
    args: argparse.Namespace,
    train_dataset: List[Dict],
    test_dataset: List[Dict],
    cfg: MathConfig,
    solver: Union[MathSolverRE, MathSolverRNN],
):
    if args.model_type == "re":
        trainer = MathTrainerRE(cfg, train_dataset, test_dataset)
    elif args.model_type == "rnn":
        trainer = MathTrainerRNN(cfg, train_dataset, test_dataset)
    else:
        raise ValueError
    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-math23k")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, test_dataset = loadMath23KRaw(args.data_path, head=args.head)
    ext_words = build_ext_words(train_dataset + test_dataset)

    const_nums = [word for word in ext_words if word not in '+-*/^()=']
    if '-1' not in const_nums:
        const_nums.append('-1')
    print(const_nums)

    train_dataset = join_Expr_list(join_const_nums(train_dataset, const_nums), args.expr_mode)
    test_dataset  = join_Expr_list(join_const_nums(test_dataset , const_nums), args.expr_mode)
    
    cfg = MathConfig(**json.loads(args.cfg))
    cfg.dataset_name = args.dataset_name
    cfg.debug = args.debug
    cfg.const_quant_size = len(const_nums)
    cfg.ext_tokens = ['^']
    
    if args.model_type == "re":
        solver = MathSolverRE(cfg)
    elif args.model_type == "rnn":
        solver = MathSolverRNN(cfg)
    else:
        raise ValueError
    
    assert args.expr_mode == "v3"
    print("expr-mode:", args.expr_mode)

    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, test_dataset, cfg, solver)


if __name__ == '__main__':
    main(get_args())
