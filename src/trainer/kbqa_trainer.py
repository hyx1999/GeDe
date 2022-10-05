import os
import re
from kbqa_utils import KBQADataset, KBClient, TrainDataInstance
from scheduler import GradualWarmupScheduler
from solver import KBQASolver
from cfg import KBQAConfig

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from loguru import logger
from typing import Dict, Any, AnyStr, List, Optional, Tuple, Union
from copy import deepcopy
from tqdm import tqdm


class KBQATrainer:

    def __init__(
        self, 
        cfg_dict: Dict[AnyStr, Any],
        dataset_dict: Dict[str, KBQADataset],
        solver: KBQASolver
    ) -> None:
        self.cfg = KBQAConfig(**cfg_dict)
        self.dataset_dict = dataset_dict
        self.train_dataset: KBQADataset = None
        self.solver = solver
        
        self.data_preprocess()

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch
    
    def train(self):
        ...
    
    def train_one_epoch(self,
        epoch: int
    ):
        ...
    
    @torch.no_grad()
    def evaluate(self):
        self.solver.eval()
        ...

    def data_preprocess(self):
        pat = re.compile("#(\d+)")
        raw_train_dataset = self.dataset_dict["train"]
        data: List[TrainDataInstance] = []
        for raw_item in tqdm(raw_train_dataset.data):
            query = raw_item.query
            expr_size = len(raw_item.exprs)
            answer = raw_item.answer
            rels     = []
            rev_rels = []
            ents_dict = {}
            for i, e in enumerate(raw_item.ents):
                ents_dict[i] = [e]
            for expr in raw_item.exprs:
                if expr.tokens[0] == "JOIN":
                    arg1 = int(pat.match(expr.tokens[1]).group(1))
                    ents_dict[expr.arg0] = self.solver.kb.execute_join(expr.tokens, ents_dict[arg1])
                    rels.append(self.solver.kb.query_rel(ents_dict[arg1], rev=False))
                    rev_rels.append(self.solver.kb.query_rel(ents_dict[arg1], rev=True))
                elif expr.tokens[0] == "AND":
                    ents_left  = None
                    ents_right = None

                elif expr.tokens[0] == "CONS":
                    ...
                elif expr.tokens[0] == "TC":
                    ...
                elif expr.tokens[0] == "ARGMAX":
                    ...
                elif expr.tokens[0] == "ARGMIN":
                    ...

            final_arg0 = raw_item.exprs[-1].arg0
            pred_ents = ents_dict[final_arg0]

        exit(0)
            # for i in range(expr_size - 1):
            #     prefix = raw_item.exprs[:i]
            #     target = raw_item.exprs[i]
            #     data.append(TrainDataInstance(query, prefix, target, rels, rev_rels))
        
        # max_rel_size = 0
        # for inst in data:
        #     max_rel_size = max(max_rel_size, len(inst.rels) + len(inst.rev_rels))
        # print("max_rel_size:", max_rel_size)
        # self.train_dataset = KBQADataset(data)
    
    @staticmethod
    def lisp_to_sparql(lisp: List[str]) -> str:
        if lisp[0] == "[JOIN]":
            ...
        ...
