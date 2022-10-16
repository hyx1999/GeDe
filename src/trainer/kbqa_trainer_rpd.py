import os
import re
from kbqa_utils import KBQADataset, DBClient, KBQADataInstance
from transformers import get_linear_schedule_with_warmup
from solver import KBQASolver
from cfg import KBQAConfig

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from loguru import logger
from typing import Dict, Any, AnyStr, List, Optional, Tuple, Union
from copy import deepcopy
from tqdm import tqdm


class KBQATrainerRPD:

    def __init__(
        self, 
        cfg: KBQAConfig,
        dataset_dict: Dict[str, List[KBQADataInstance]],
        solver: KBQASolver,
        use_dev: bool = True
    ) -> None:
        self.cfg = cfg
        self.raw_dataset = dataset_dict
        self.solver = solver
        self.use_dev = use_dev
        self.best_ranker_dev_acc  = None
        self.best_ranker_test_acc = None
        self.best_dev_acc  = None
        self.best_test_acc = None

        rel_set = set(solver.all_relations)
        tp_set  = set(solver.all_types)
        rel2id = {r: index for index, r in enumerate(solver.all_relations)}
        tp2id  = {t: index for index, t in enumerate(solver.all_types)}
        for value in dataset_dict.values():
            for item in value:
                item.parse_target_relation_mask(rel2id)
                item.parse_target_type_mask(tp2id)
                item.tag = \
                    (len(rel_set & set(item.answer_relations)) == len(item.answer_relations)) and \
                    (len(tp_set  & set(item.answer_types    )) == len(item.answer_types    ))

        if use_dev and "dev" not in dataset_dict:
            self.split_data()
        self.train_dataset = self.convert_dataset(self.raw_dataset["train"])

    def convert_dataset(self, dataset: List[KBQADataInstance]) -> List[KBQADataInstance]:
        return dataset

    def split_data(self):
        raw_train_dataset = self.raw_dataset["train"]
        val_ids = set(np.random.choice(len(raw_train_dataset), size=100, replace=False).tolist())
        train_dataset = []
        dev_dataset   = []
        for i in range(len(raw_train_dataset)):
            if i in val_ids:
                dev_dataset.append(raw_train_dataset[i])
            else:
                train_dataset.append(raw_train_dataset[i])
        self.raw_dataset["train"] = train_dataset
        self.raw_dataset["dev"] = dev_dataset

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch
    
    def train_ranker(self):
        self.solver.to(self.cfg.device)
        
        dataset = KBQADataset(self.train_dataset)
        shuffle_flag = not self.cfg.debug
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle_flag, collate_fn=self.collate_fn)

        optim = AdamW(
            [
                {'params': self.solver.ranker.parameters(), 'lr': self.cfg.lr},
            ],
            weight_decay=self.cfg.weight_decay
        )

        print("num_training_steps = {}".format(self.cfg.num_epochs * len(loader)))

        scheduler = get_linear_schedule_with_warmup(
            optim, 
            num_warmup_steps=0,
            num_training_steps=self.cfg.num_epochs * len(loader)
        )
        
        for epoch in range(self.cfg.num_epochs):                
            self.train_ranker_one_epoch(epoch, self.solver, optim, scheduler, loader)
            
            if epoch > self.cfg.num_epochs // 2 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
                if not self.cfg.debug:
                    if self.use_dev:
                        logger.info("[evaluate dev-data]")
                        dev_ranker_acc = self.evaluate_ranker("dev", epoch, self.solver, self.raw_dataset["dev"])
                    logger.info("[evaluate test-data]")
                    test_ranker_acc = self.evaluate_ranker("test", epoch, self.solver, self.raw_dataset["test"])
                    
                    if not self.use_dev:
                        dev_ranker_acc = test_ranker_acc

                    if self.best_ranker_dev_acc is None or dev_ranker_acc >= self.best_ranker_dev_acc:
                        self.best_ranker_dev_acc  = dev_ranker_acc
                        self.best_ranker_test_acc = test_ranker_acc
                else:
                    logger.info("[evaluate train-data]")
                    self.evaluate_ranker("train", epoch, self.solver, self.raw_dataset["train"][:100])

    def train_ranker_one_epoch(self,
        epoch: int,
        solver: KBQASolver,
        optim: Union[Adam, AdamW],
        scheduler: LambdaLR,
        loader: DataLoader
    ):
        ...

    @torch.no_grad()
    def evaluate_ranker(self,
        dataset_type: str,
        epoch: int,
        solver: KBQASolver,
        test_data: List[Dict]    
    ):
        self.solver.eval()
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
