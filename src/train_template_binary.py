from dataset import loadTemplate
from solver import MathSolverSeq2Seq
from trainer import MathTrainerSeq2Seq
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
    dev_dataset: List[Dict],
    test_dataset: List[Dict],
    cfg: MathConfig,
    solver: MathSolverSeq2Seq,
):
    trainer = MathTrainerSeq2Seq(cfg, train_dataset, test_dataset, dev_dataset=dev_dataset)

    trainer.train(solver)
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-linalg")
    logger.info("[finish train solver]")
    logger.info("best test acc: {}".format(trainer.best_test_acc))


def main(args: argparse.Namespace):
    # if not args.debug:
    setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    
    train_dataset, dev_dataset, test_dataset = loadTemplate(args.data_path, head=args.head)
    
    cfg = MathConfig(**json.loads(args.cfg))
    cfg.dataset_name = args.dataset_name
    cfg.debug = args.debug
    cfg.const_quant_size = 4
    cfg.ext_tokens = ["^"]
    
    logger.info("len(const_quant_size): {}".format(0))
    logger.info("const_quants: {}".format([]))

    solver = MathSolverSeq2Seq(cfg)
    
    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, train_dataset, dev_dataset, test_dataset, cfg, solver)


if __name__ == '__main__':
    main(get_args())
