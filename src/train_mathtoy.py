from dataset import loadMathToy
from solver import MathSolver
from trainer import MathTrainer
from math_utils import Expr
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
    cfg: MathConfig,
    solver: MathSolver,
):
    trainer = MathTrainer(cfg, train_dataset, test_dataset)
    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-mathtoy")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, test_dataset = loadMathToy(args.data_path, head=args.head)
    const_nums = []

    cfg = MathConfig(**json.loads(args.cfg))
    cfg.dataset_name = args.dataset_name
    cfg.debug = args.debug
    solver = MathSolver(cfg, const_nums)
    
    if args.expr_mode == "v1":
        for obj in train_dataset:
            obj["Expr_list"] = obj["Expr_list0"]
        for obj in test_dataset:
            obj["Expr_list"] = obj["Expr_list0"]
    else:
        for obj in train_dataset:
            obj["Expr_list"] = [obj["Expr_list1"]]
        for obj in test_dataset:
            obj["Expr_list"] = [obj["Expr_list1"]]
    
    for dataset in [train_dataset, test_dataset]:
        for obj in dataset:
            Expr_list = [
                Expr(
                    arg0=opseq_obj["arg0"],
                    expr_toks=opseq_obj["expr_toks"],
                    expr_str="".join(opseq_obj["expr_toks"])
                ) for opseq_obj in obj["Expr_list"]
            ]
            obj["Expr_list"] = Expr_list

    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, test_dataset, cfg, solver)


if __name__ == '__main__':
    main(get_args())
