from data_process import loadMath23K, build_ext_words, join_constant_nums, join_OpSeq_list
from solver import RecursionSolver
from trainer import RecursionTrainer

import datetime
import argparse
import json
from loguru import logger
from typing import Dict, List, Union
from copy import deepcopy
from tqdm import tqdm


def setup_logger():
    format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    format="[{time}]-[{level}] {file}-{function}-{line}: {message}"
    logger.remove(None)
    logger.add(
        f"log/{format_time}.log",
        rotation="100 MB", 
        level="INFO", 
        format=format
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_text", type=str, default="none")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--load_model_dir", type=str, required=True)
    parser.add_argument("--save_model_dir", type=str, required=True)
    parser.add_argument("--head", type=int, default=-1)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cfg", type=str, default="{}")
    parser.add_argument("--save_model", action="store_true")
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
        solver.save_model(args.save_model_dir, "final")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    setup_logger()

    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, test_dataset = loadMath23K(args.data_path, args.fold, head=args.head)
    ext_words = build_ext_words(train_dataset)

    constant_nums = [word for word in ext_words if word not in '+-*/^()=']
    
    for dataset in [train_dataset, test_dataset]:
        join_constant_nums(dataset, constant_nums)
        join_OpSeq_list(dataset)
    
    train_dataset = [obj for obj in train_dataset if "OpSeq_list" in obj]
    test_dataset = [obj for obj in test_dataset if "OpSeq_list" in obj]

    solver = RecursionSolver(json.loads(args.cfg))

    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, test_dataset, solver)


if __name__ == '__main__':
    main(get_args())
