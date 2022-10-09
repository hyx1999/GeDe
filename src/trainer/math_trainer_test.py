import os
import math
from solver import MathSolverTest
from scheduler import GradualWarmupScheduler
from math_utils import DefaultDataset, compute_Expr_list
from cfg import MathConfig
from math_utils import ExprDataInstance

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


class MathTrainerTest:

    def __init__(
        self, 
        cfg: MathConfig,
        train_dataset: List[Dict[AnyStr, Any]],
        test_dataset: List[Dict[AnyStr, Any]],
        use_dev: bool = True
    ) -> None:
        self.cfg = cfg
        self.use_dev = use_dev
        self.raw_dataset = {
            "train": deepcopy(train_dataset),
            "test": deepcopy(test_dataset),
        }
        if self.use_dev:
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

    def convert_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[ExprDataInstance]:
        new_dataset = []
        for obj in dataset:
            question = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            expr_list = obj["Expr_list"]
            for i in range(len(expr_list)):
                new_dataset.append(ExprDataInstance(
                    question=question,
                    nums=nums,
                    const_nums=const_nums,
                    expr_list=expr_list[:i],
                    target=[expr_list[i]],
                    id=obj["sample_id"],
                    end=(i + 1 == len(expr_list))
                ))
        
        if "svamp" in self.cfg.dataset_name:
            new_dataset.extend(self.augment_dataset(dataset))
        return new_dataset

    def augment_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[ExprDataInstance]:
        return []

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: MathSolverTest):
        solver.to(self.cfg.device)

        optim = AdamW(
            [
                {'params': solver.encoder.parameters(), 'lr': self.cfg.bert_lr},
                {'params': solver.decoder.parameters(), 'lr': self.cfg.gru_lr},
            ],
            weight_decay=self.cfg.weight_decay
        )

        scheduler_steplr = StepLR(optim, step_size=self.cfg.scheduler_step_size, gamma=0.5)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_steplr)

        dataset = DefaultDataset(self.train_dataset)
        shuffle_flag = not self.cfg.debug
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle_flag, collate_fn=self.collate_fn)

        for epoch in range(self.cfg.num_epochs):
            self.train_one_epoch(epoch, solver, optim, loader)
            scheduler_warmup.step()
            
            if epoch > 0 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
                if not self.cfg.debug:
                    if self.use_dev:
                        logger.info("[evaluate dev-data]")
                        dev_acc = self.evaluate("dev", epoch, solver, self.raw_dataset["dev"])
                    logger.info("[evaluate test-data]")
                    test_acc = self.evaluate("test", epoch, solver, self.raw_dataset["test"])
                    
                    if not self.use_dev:
                        dev_acc = test_acc
                    
                    if epoch >= 40 and (self.best_dev_acc is None or dev_acc >= self.best_dev_acc):
                        self.best_dev_acc = dev_acc
                        self.best_test_acc = test_acc
                else:
                    logger.info("[evaluate train-data]")
                    self.evaluate("train", epoch, solver, self.raw_dataset["train"][:100])


    def train_one_epoch(
        self,
        epoch: int,
        solver: MathSolverTest,
        optim: Union[Adam, AdamW],
        loader: DataLoader
    ) -> None:
        solver.train()

        pbar = tqdm(loader, desc="recursion-train", total=len(loader))

        loss_total = 0
        mAcc = 0
        for i, batch in enumerate(pbar):            
            if i < 5 and epoch == 0:
                for x in [
                    I.parse_input(solver.lang_tok.sep_token) \
                    + " ---> " \
                    + I.parse_output(solver.expr_tok.bos_token, solver.expr_tok.eos_token) for I in batch
                ]:
                    print(x)

            loss, Acc = solver(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_total += loss.item()
            mAcc += Acc.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        mAcc = mAcc / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}, mAcc: {mAcc}")

# --------------------------------------------- evaluate ------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataset_type: str,
        epoch: int,
        solver: MathSolverTest,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        Acc  = []

        test_dataset = DefaultDataset(test_data)
        
        if self.cfg.save_result:
            os.makedirs("../cache/mwp", exist_ok=True)
            f = open("../cache/mwp/{}_{}_{}.txt".format(self.cfg.dataset_name, dataset_type, epoch), "w")
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            # if self.cfg.debug:
            #     print(">")
            #     print("target:", [" ".join(x.expr_toks) for x in obj["Expr_list"]])
            # output_Expr_list = solver.generate(input_text, nums, const_nums)
            output_Expr_list = solver.beam_search(input_text, nums, const_nums, beam_size=self.cfg.beam_size)
            target_Expr_list = obj["Expr_list"]

            try:
                output_value = compute_Expr_list(output_Expr_list, nums, const_nums, self.cfg.max_nums_size)
                target_value = compute_Expr_list(target_Expr_list, nums, const_nums, self.cfg.max_nums_size)
            except SyntaxError:
                output_value = None
                target_value = None
            eps = 1e-5

            # TEST USE
            # if i < 20:
            #     logger.info("i: {}".format(i))
            #     logger.info("input_text: {}".format(input_text))
            #     logger.info("output_Op_list: {}".format(output_Expr_list))
            #     logger.info("target_Op_list: {}".format(target_Expr_list))
            #     logger.info("output_value: {}".format(output_value))
            #     logger.info("target_value: {}".format(target_value))

            if (output_value is not None and target_value is not None and abs(output_value - target_value) < eps):
                Acc.append(1)
            else:
                Acc.append(0)
            
            expr_list0 = [" ".join(x.expr_toks) for x in output_Expr_list] if output_Expr_list is not None else None
            expr_list1 = [" ".join(x.expr_toks) for x in target_Expr_list] if target_Expr_list is not None else None
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