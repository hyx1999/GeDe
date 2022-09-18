from data_process import loadMath23K, build_ext_words, join_const_nums, join_OpSeq_list
from solver import RecursionSolver
from trainer import RecursionTrainer

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
    parser.add_argument("--log_text", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--load_model_dir", type=str, required=True)
    parser.add_argument("--save_model_dir", type=str, required=True)
    parser.add_argument("--head", type=int, default=-1)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cfg", type=str, default="{}")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--op_seq_mode", type=str, default="v1")
    return parser.parse_args()


def train_solver(
    args: argparse.Namespace,
    train_dataset: List[Dict],
    test_dataset: List[Dict],
    solver: RecursionSolver,
):
    cfg = json.loads(args.cfg)
    trainer = RecursionTrainer(cfg, train_dataset, test_dataset)
    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-20220917")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, test_dataset = loadMath23K(args.data_path, args.fold, head=args.head)
    ext_words = build_ext_words(train_dataset + test_dataset)

    const_nums = [word for word in ext_words if word not in '+-*/^()=']
    if '-1' not in const_nums:
        const_nums.append('-1')
    print(const_nums)

    train_dataset = join_OpSeq_list(join_const_nums(train_dataset, const_nums), args.op_seq_mode)
    test_dataset  = join_OpSeq_list(join_const_nums(test_dataset , const_nums), args.op_seq_mode)
    
    solver = RecursionSolver(json.loads(args.cfg), const_nums)
    
    solver.cfg.op_seq_mode = args.op_seq_mode
    if args.op_seq_mode == "v2":
        solver.cfg.max_step_size = 1
        solver.cfg.use_bracket = True
    print(args.op_seq_mode)


    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, test_dataset, solver)


if __name__ == '__main__':
    main(get_args())
