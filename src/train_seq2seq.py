from data_process import loadMath23K, build_ext_words
from solver import Seq2seqSolver
from trainer import Seq2seqTrainer

import datetime
import argparse
import json
from loguru import logger
from typing import Dict, List, Union

import torch
import numpy as np
import random

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
    parser.add_argument("--log_text", type=str, default="none")
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

def train_math_solver(
    args: argparse.Namespace,
    train_dataset: List[Dict],
    test_dataset: List[Dict],
    solver: Seq2seqSolver
):
    cfg = json.loads(args.cfg)
    trainer = Seq2seqTrainer(cfg, train_dataset, test_dataset)
    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "baseline")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    setup_logger()
    setup_seed()
    
    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, test_dataset = loadMath23K(args.data_path, args.fold, head=args.head)
    ext_words = build_ext_words(train_dataset)

    print("len(ext_words):", len(ext_words))
    print("ext_words:", ext_words)

    solver = Seq2seqSolver(json.loads(args.cfg), ext_words)

    if args.save_model:
        solver.save_model("models_test")

    train_math_solver(args, train_dataset, test_dataset, solver)


if __name__ == '__main__':
    main(get_args())
