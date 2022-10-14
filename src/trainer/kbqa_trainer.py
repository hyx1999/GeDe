import os
import re
from kbqa_utils import KBQADataset, DBClient, KBQADataInstance
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
        dataset_dict: Dict[str, List[Dict[str, Any]]],
        solver: KBQASolver
    ) -> None:
        self.cfg = KBQAConfig(**cfg_dict)
        self.dataset_dict = dataset_dict
        self.train_dataset: KBQADataset = None
        self.solver = solver

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch
    
    def train_ranker(self):
        ...
    
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
