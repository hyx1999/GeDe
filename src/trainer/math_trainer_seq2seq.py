import os
import math
import random
from solver import MathSolverSeq2Seq
from scheduler import GradualWarmupScheduler
from math_utils import MathDataset, compute_PreOrder
from cfg import MathConfig
from math_utils import MathDataInstance
from transformers import get_linear_schedule_with_warmup

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import Dataset, DataLoader

from loguru import logger
from typing import Dict, Any, AnyStr, List, Optional, Tuple, Union
from copy import deepcopy
from tqdm import tqdm


class MathTrainerSeq2Seq:

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
        
        # self.evaluate_test(self.raw_dataset["test"])

    def convert_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[MathDataInstance]:
        new_dataset = []
        for obj in dataset:
            question = " ".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            expr_list = obj["Expr_list"]
            new_dataset.append(MathDataInstance(
                question=question,
                nums=nums,
                const_nums=const_nums,
                expr_list=expr_list,
                target=expr_list,
                id=obj["sample_id"]
            ))

        return new_dataset

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: MathSolverSeq2Seq):
        solver.to(self.cfg.device)
        
        dataset = MathDataset(self.train_dataset)
        shuffle_flag = not self.cfg.debug
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle_flag, collate_fn=self.collate_fn)

        param_dict = {
            "encoder": [],
            "decoder": [],
            "encoder_no_decay": [],
            "decoder_no_decay": [],
        }
        # no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        no_decay = ["LayerNorm"]
        for name, p in solver.named_parameters():
            if "encoder" in name:
                if any(nd in name for nd in no_decay):
                    param_dict["encoder_no_decay"].append(p)
                else:
                    param_dict["encoder"].append(p)
            elif "decoder" in name:
                if any(nd in name for nd in no_decay):
                    param_dict["decoder_no_decay"].append(p)
                else:
                    param_dict["decoder"].append(p)
            else:
                print("name: {}".format(name))
                raise ValueError

        alpha = self.cfg.lr_alpha
        optim = AdamW(
            [
                {'params': param_dict["encoder"] , 'lr': self.cfg.lr        , 'weight_decay': self.cfg.weight_decay},
                {'params': param_dict["decoder"] , 'lr': self.cfg.lr * alpha, 'weight_decay': self.cfg.weight_decay},
                {'params': param_dict["encoder_no_decay"], 'lr': self.cfg.lr        , 'weight_decay': 0.0},
                {'params': param_dict["decoder_no_decay"], 'lr': self.cfg.lr * alpha, 'weight_decay': 0.0},
            ],
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optim, 
            num_warmup_steps=0,
            num_training_steps=self.cfg.num_epochs * len(loader)
        )

        for epoch in range(self.cfg.num_epochs):

            self.train_one_epoch(epoch, solver, optim, scheduler, loader)
            
            if epoch > 0 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
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
        solver: MathSolverSeq2Seq,
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
                    I.parse_input(sep_token="", use_expr=False) \
                    + " ---> " \
                    + I.parse_output(solver.expr_tok.bos_token, solver.expr_tok.eos_token) for I in batch
                ]:
                    print(x)

            loss, Acc = solver(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            loss_total += loss.item()
            mAcc += Acc.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        mAcc = mAcc / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}, mAcc: {mAcc}")

# --------------------------------------------- evaluate ------------------------------------------------------

    @torch.no_grad()
    def evaluate_test(
        self,
        test_data: List[Dict]
    ) -> float:        

        test_dataset = MathDataset(test_data)
                
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            target_PreOrder = obj["Expr_list"][0].expr_toks

            try:
                target_value = compute_PreOrder(target_PreOrder, nums, const_nums, self.cfg.quant_size)
            except SyntaxError:
                target_value = None
            print(target_value)

        exit(0)


    @torch.no_grad()
    def evaluate(
        self,
        dataset_type: str,
        epoch: int,
        solver: MathSolverSeq2Seq,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        Acc  = []

        test_dataset = MathDataset(test_data)
        
        if self.cfg.save_result:
            os.makedirs("../cache/mwp", exist_ok=True)
            f = open("../cache/mwp/{}_{}_{}_{}.txt".format(self.cfg.dataset_name, dataset_type, epoch, solver._get_name()), "w")
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = " ".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            output_PreOrder = solver.beam_search(input_text, nums, const_nums, beam_size=self.cfg.beam_size)
            target_PreOrder = obj["Expr_list"][0].expr_toks

            try:
                output_value = compute_PreOrder(output_PreOrder, nums, const_nums, self.cfg.quant_size)
                target_value = compute_PreOrder(target_PreOrder, nums, const_nums, self.cfg.quant_size)
            except SyntaxError:
                output_value = None
                target_value = None
            eps = 1e-5

            if (output_value is not None and target_value is not None and abs(output_value - target_value) < eps):
                Acc.append(1)
            else:
                Acc.append(0)
            
            expr_list0 = output_PreOrder
            expr_list1 = target_PreOrder
            if self.cfg.save_result:
                f.write("id: {}\n".format(i))
                f.write("input={}\n".format(input_text))
                f.write("nums={}\n".format(nums))
                f.write("output={}\n".format(expr_list0))
                f.write("target={}\n".format(expr_list1))
                f.write("correct={}\n".format("True" if Acc[-1] == 1 else "False"))

        if self.cfg.save_result:
            f.close()

        answer_mAcc = sum(Acc) / len(Acc)
        msg = "epoch: {} answer-mAcc: {}".format(epoch, answer_mAcc)
        logger.info(msg)

        return answer_mAcc
