from dataset import loadMathQA
from solver import MathSolverRNN, MathSolverRE, MathSolverRE_Abl0
from trainer import MathTrainerRNN, MathTrainerRE, MathTrainerRE_Abl0
from cfg import MathConfig
from torchstat import stat

import os
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
    parser.add_argument("--cfg", type=str, default="{}")
    return parser.parse_args()


def main(args: argparse.Namespace):
    
    cfg = MathConfig(**json.loads(args.cfg))
    cfg.const_quant_size = 19
    cfg.ext_tokens = ['^']
    
    solver_dict = {
        "re": MathSolverRE,
        "re_abl0": MathSolverRE_Abl0,
        "rnn": MathSolverRNN,
    }
    if args.model_type in solver_dict:
        solver: torch.nn.Module = solver_dict[args.model_type](cfg)
    else:
        raise ValueError(args.model_type)

    num_params = 0
    for param in solver.parameters():
        num_params += param.numel()
    print(num_params, num_params / 1024 / 1024)


if __name__ == '__main__':
    main(get_args())
