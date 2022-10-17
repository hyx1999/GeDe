import os
import random
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
        self.best_ranker_dev_recall  = None
        self.best_ranker_test_recall = None
        self.best_dev_acc  = None
        self.best_test_acc = None

        rel_set = set(solver.all_relations)
        tp_set  = set(solver.all_types)
        for value in dataset_dict.values():
            for item in value:
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
    
    def add_negative(self, part: List[str], all: List[str], num: int):
        candidates = list(set(all) - set(part))
        ids = np.random.choice(len(candidates), size=num - len(part), replace=False).tolist()
        result = part + [candidates[i] for i in ids]
        random.shuffle(result)
        return result
    
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
                        dev_ranker_recall = self.evaluate_ranker("dev", epoch, self.solver, self.raw_dataset["dev"])
                    logger.info("[evaluate test-data]")
                    test_ranker_recall = self.evaluate_ranker("test", epoch, self.solver, self.raw_dataset["test"])
                    
                    if not self.use_dev:
                        dev_ranker_recall = test_ranker_recall

                    if self.best_ranker_dev_recall is None or dev_ranker_recall >= self.best_ranker_dev_recall:
                        self.best_ranker_dev_recall  = dev_ranker_recall
                        self.best_ranker_test_recall = test_ranker_recall
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
        NUM_RELATION = 256
        NUM_TYPES = 256
        solver.train()

        pbar = tqdm(loader, desc="ranker-train", total=len(loader))

        loss_total = 0
        for i, batch in enumerate(pbar):
            batch: List[KBQADataInstance] = batch
            if i < 5 and epoch == 0:
                for x in [
                    I.parse_input() \
                    + " ---> " \
                    + I.parse_output(solver.logic_tokenizer.bos_token, solver.logic_tokenizer.eos_token) for I in batch
                ]:
                    print(x)

            rels = list(set(sum([I.answer_relations for I in batch], [])))
            rels = self.add_negative(rels, solver.all_relations, min(NUM_RELATION, len(solver.all_relations)))
            rel2id = {r: index for index, r in enumerate(rels)}
            
            tps  = list(set(sum([I.answer_types for I in batch], [])))
            tps  = self.add_negative(tps, solver.all_types, min(NUM_TYPES, len(solver.all_types)))
            tp2id = {t: index for index, t in enumerate(tps)}
            
            inputs  = [I.parse_input() for I in batch]
            rel_target = [I.parse_target_relation_mask(rel2id) for I in batch]
            tp_target  = [I.parse_target_type_mask(tp2id) for I in batch]
            
            text_input = solver.lang_tokenizer(inputs, padding=True, return_tensors="pt").to(self.cfg.device)
            rel_input  = solver.lang_tokenizer(rels  , padding=True, return_tensors="pt").to(self.cfg.device)
            tp_input   = solver.lang_tokenizer(tps   , padding=True, return_tensors="pt").to(self.cfg.device)
            rel_target = torch.tensor(rel_target, dtype=torch.float, device=self.cfg.device)
            tp_target  = torch.tensor(tp_target , dtype=torch.float, device=self.cfg.device)

            rel_loss: Tensor = solver.ranker(text_input, rel_input, rel_target)
            tp_loss: Tensor  = solver.ranker(text_input, tp_input , tp_target )
            loss = rel_loss + tp_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            loss_total += loss.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}")


    @torch.no_grad()
    def evaluate_ranker(self,
        dataset_type: str,
        epoch: int,
        solver: KBQASolver,
        test_data: List[KBQADataInstance]
    ):
        self.solver.eval()
        recall_relation = []
        recall_type     = []
        
        if self.cfg.save_result:
            os.makedirs("../cache/kbqa", exist_ok=True)
            f = open("../cache/kbqa/{}_{}_{}_ranker.txt".format(self.cfg.dataset_name, dataset_type, epoch), "w")

        RELATION_K = min(100, len(solver.all_relations))
        TYPE_K = min(100, len(solver.all_types))
        
        for i in tqdm(range(len(test_data)), desc="evaluate", total=len(test_data)):
            item = test_data[i]
            candidate_relations, candidate_types = solver.rank(item.query, RELATION_K, TYPE_K)
            recall_relation.append((set(candidate_relations) & set(item.answer_relations)) / len(item.answer_relations))
            recall_type.append((set(candidate_types) & set(item.answer_types)) / len(item.answer_types))
            
        if self.cfg.save_result:
            f.close()
        
        mean_recall_relation = sum(recall_relation) / len(recall_relation)
        mean_recall_type     = sum(recall_type)     / len(recall_type    )
        
        msg = "epoch: {} relation-mRecall: {} type-mRecall: {}".format(epoch, mean_recall_relation, mean_recall_type)
        logger.info(msg)
        
        return mean_recall_relation


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
