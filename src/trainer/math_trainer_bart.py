import json
import os
import re
import math
import random
from solver import MathSolverBART
from scheduler import GradualWarmupScheduler
from math_utils import MathDataset, compute_Expr_list, compute_MultiExpr_list
from cfg import MathConfig
from math_utils import MathDataInstance, TemplateDataInstance, Expr, MultiExpr
from transformers import get_linear_schedule_with_warmup, GenerationConfig

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


class MathTrainerBART:

    def __init__(
        self, 
        cfg: MathConfig,
        train_dataset: List[Dict[AnyStr, Any]],
        test_dataset: List[Dict[AnyStr, Any]],
        dev_dataset: Optional[Dict[AnyStr, Any]] = None,
        use_dev: bool = True
    ) -> None:
        self.cfg = cfg
        self.use_dev = use_dev
        self.raw_dataset = {
            "raw_train": deepcopy(train_dataset),
            "train": deepcopy(train_dataset),
            "test": deepcopy(test_dataset),
            "dev": deepcopy(dev_dataset),
        }
        
        if self.use_dev and dev_dataset is None:
            self.split_data()
        self.train_dataset = self.convert_dataset(self.raw_dataset["train"])
        self.best_dev_acc = None
        self.best_test_acc = None

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

    def convert_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[MathDataInstance]:
        new_dataset = []
        for obj in dataset:
            question = " ".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            expr_list = obj["Expr_list"]
            new_dataset.append(TemplateDataInstance(
                question=question,
                nums=nums,
                const_nums=const_nums,
                expr_list=[],
                target=expr_list,
                id=obj["sample_id"],
                end=True
            ))
        return new_dataset

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: MathSolverBART):
        solver.to(self.cfg.device)
        
        dataset = MathDataset(self.train_dataset)
        shuffle = not self.cfg.debug
        loader = DataLoader(
            dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=shuffle, 
            collate_fn=self.collate_fn
        )

        param_dict = {
            "decay": [],
            "no_decay": [],
        }
        no_decay = ["LayerNorm"]
        for name, p in solver.named_parameters():
            if any(nd in name for nd in no_decay):
                param_dict["no_decay"].append(p)
            else:
                param_dict["decay"].append(p)

        alpha = self.cfg.lr_alpha
        optim = AdamW(
            [
                {
                    'params': param_dict["decay"] , 
                    'lr': self.cfg.lr, 
                    'weight_decay': self.cfg.weight_decay
                },
                {
                    'params': param_dict["no_decay"], 
                    'lr': self.cfg.lr, 
                    'weight_decay': 0.0
                },            
            ],
        )

        print("num_training_steps = {}".format(self.cfg.num_epochs * len(loader)))

        scheduler = get_linear_schedule_with_warmup(
            optim, 
            num_warmup_steps=0,
            num_training_steps=self.cfg.num_epochs * len(loader)
        )
        
        for epoch in range(self.cfg.num_epochs):                
            self.train_one_epoch(epoch, solver, optim, scheduler, loader)
            
            if (epoch > self.cfg.num_epochs // 2 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5) \
                or (epoch == 0):
                if not self.cfg.debug:
                    if self.use_dev:
                        logger.info("[evaluate dev-data]")
                        dev_acc = self.evaluate("dev", epoch, solver, self.raw_dataset["dev"])
                    logger.info("[evaluate test-data]")
                    test_acc = self.evaluate("test", epoch, solver, self.raw_dataset["test"])
                    
                    if not self.use_dev:
                        dev_acc = test_acc
                    
                    if self.best_dev_acc is None or dev_acc >= self.best_dev_acc:
                        self.best_dev_acc = dev_acc
                        self.best_test_acc = test_acc
                else:
                    logger.info("[evaluate train-data]")
                    self.evaluate("train", epoch, solver, self.raw_dataset["train"][:100])


    def train_one_epoch(
        self,
        epoch: int,
        solver: MathSolverBART,
        optim: Union[Adam, AdamW],
        scheduler: LambdaLR,
        loader: DataLoader
    ) -> None:
        solver.train()

        pbar = tqdm(loader, desc="recursion-train", total=len(loader))

        loss_total = 0
        mAcc = 0
        for i, batch in enumerate(pbar):            
            if i < 5 and epoch == 0:
                for x in [
                    I.parse_input("#") \
                    + " ---> " \
                    + I.parse_output_bart(solver.tok.bos_token, solver.tok.eos_token) for I in batch
                ]:
                    print(x)
            
            loss, acc_score = solver(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            loss_total += loss.item()
            mAcc += acc_score.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        mAcc = mAcc / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}, mAcc: {mAcc}")

# --------------------------------------------- evaluate ------------------------------------------------------

    def parse_num(self, x: str) -> int:
        mt = re.match("\[num(\d+)\]", x)
        if mt is None:
            raise SyntaxError
        return int(mt.group(1))
    
    def parse_expr_list(self, 
        solver: MathSolverBART,
        output: torch.LongTensor,
    ) -> List[MultiExpr]:

        exprs = solver.tok.decode(output)\
            .replace(solver.tok.eos_token, "")\
            .strip()\
            .split(solver.tok.bos_token)
        
        expr_list = []
        for i, expr_stat in enumerate(exprs):
            if expr_stat.strip() == "":
                continue
            if "[->]" in expr_stat and expr_stat.count("[->]") == 1:
                expr_str, args_str = expr_stat.split("[->]")
                args = [self.parse_num(x) for x in args_str.strip().split(" ") if x.strip() != ""]
                expr_toks = expr_str.strip().split(" ")
                expr_list.append(MultiExpr(args=args, expr_toks=expr_toks, expr_str=expr_str.strip()))
            else:
                raise SyntaxError

        return expr_list

    
    @torch.no_grad()
    def evaluate(
        self,
        dataset_type: str,
        epoch: int,
        solver: MathSolverBART,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        Acc  = []

        test_dataset = MathDataset(test_data)
        g_config = GenerationConfig(max_new_tokens=100 if epoch > 1 else 20)
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = " ".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]

            inputs = solver.tok(input_text, return_tensors="pt")\
                .input_ids.to(self.cfg.device)
            
            output = solver\
                .model\
                .generate(inputs, g_config)\
                .squeeze(0)
            
            try:
                output_Expr_list = self.parse_expr_list(solver, output)
            except SyntaxError:
                output_Expr_list = None

            try:
                output_value = compute_MultiExpr_list(output_Expr_list, nums, const_nums, self.cfg.quant_size)
            except SyntaxError:
                output_value = None
            except IndexError:
                output_value = None                

            target_Expr_list = obj["Expr_list"]
            try:
                target_value = compute_MultiExpr_list(target_Expr_list, nums, const_nums, self.cfg.quant_size)
            except SyntaxError:
                target_value = None
            except IndexError:
                target_value = None                
            
            if target_value is None:
                continue

            eps = 1e-5

            if (output_value is not None and abs(output_value - target_value) < eps):
                Acc.append(1)
            else:
                Acc.append(0)

        answer_mAcc = sum(Acc) / (len(Acc) + 1e-5)
        msg = "epoch: {} answer-mAcc: {}".format(epoch, answer_mAcc)
        logger.info(msg)

        return answer_mAcc
