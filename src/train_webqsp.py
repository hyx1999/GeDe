from dataset import loadWebQSP
from solver import KBQASolver
from trainer import KBQATrainer
from kbqa_utils import build_extra_tokens, build_domain_info

import datetime
import argparse
import json
from loguru import logger
from typing import Dict, List, Set, Union
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
    parser.add_argument("--head", type=int, default=-1)
    parser.add_argument("--cfg", type=str, default="{}")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def train_solver(
    args: argparse.Namespace,
    dataset_dict: List[Dict],
    solver: KBQASolver,
):
    cfg = json.loads(args.cfg)
    trainer = KBQATrainer(cfg, dataset_dict, solver)
    trainer.cfg.dataset_name = args.dataset_name
    trainer.cfg.debug = args.debug
    trainer.train()
    if args.save_model:
        solver.save_model(args.save_model_dir, "final-v3")
    logger.info("[finish train solver]")


def main(args: argparse.Namespace):
    if not args.debug:
        setup_logger()
    setup_seed()
    logger.info("log_text: {}".format(args.log_text))
    
    dataset_dict, rel_dict = loadWebQSP(args.data_path, head=args.head)    
    extra_tokens = build_extra_tokens(dataset_dict, rel_dict)
    domain_info = build_domain_info(rel_dict)
    solver = KBQASolver(
        json.loads(args.cfg),
        extra_tokens,
        domain_info
    )
    
    if args.save_model:
        solver.save_model(args.save_model_dir, "test")
    
    train_solver(args, dataset_dict, solver)


if __name__ == '__main__':
    main(get_args())
