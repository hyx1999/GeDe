import os
from solver import MathSolver
from scheduler import GradualWarmupScheduler
from math_utils import DefaultDataset, compute_OpSeq_list
from cfg import MathConfig
from math_utils import OpSeqDataInstance

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


class MathTrainer:

    def __init__(
        self, 
        cfg_dict: Dict[AnyStr, Any],
        train_dataset: List[Dict[AnyStr, Any]],
        test_dataset: List[Dict[AnyStr, Any]]
    ) -> None:
        self.cfg = MathConfig(**cfg_dict)
        self.raw_dataset = {
            "train": deepcopy(train_dataset),
            "test": deepcopy(test_dataset),
        }
        self.split_data()
        self.train_dataset = self.convert_dataset(self.raw_dataset["train"])
    
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
        
    def convert_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[OpSeqDataInstance]:
        new_dataset = []
        for obj in dataset:
            question = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            OpSeq_list = obj["OpSeq_list"]
            new_dataset.append(OpSeqDataInstance(
                question=question,
                nums=nums,
                const_nums=const_nums,
                opSeq_list=OpSeq_list[:-1],
                target=OpSeq_list,
                id=obj["sample_id"]
            ))

        return new_dataset

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: MathSolver):
        solver.to(self.cfg.device)

        optim = AdamW(
            [
                {'params': solver.encoder.parameters(), 'lr': 5e-5},
                {'params': solver.decoder.parameters(), 'lr': 5e-4},
            ],
            weight_decay=1e-4
        )

        scheduler_steplr = StepLR(optim, step_size=self.cfg.scheduler_step_size, gamma=0.5)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_steplr)

        dataset = DefaultDataset(self.train_dataset)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=self.collate_fn)

        for epoch in range(self.cfg.num_epochs):
            self.train_one_epoch(epoch, solver, optim, loader)
            scheduler_warmup.step()
            
            if epoch > 0 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
                if not self.cfg.debug:
                    logger.info("[evaluate dev-data]")
                    self.evaluate(epoch, solver, self.raw_dataset["dev"])
                    logger.info("[evaluate test-data]")
                    self.evaluate(epoch, solver, self.raw_dataset["test"])
                else:
                    logger.info("[evaluate train-data]")
                    self.evaluate(epoch, solver, self.raw_dataset["train"][:20])


    def train_one_epoch(
        self,
        epoch: int,
        solver: MathSolver,
        optim: Union[Adam, AdamW],
        loader: DataLoader
    ) -> None:
        solver.train()

        pbar = tqdm(loader, desc="recursion-train", total=len(loader))

        loss_total = 0
        mAcc = 0
        for i, batch in enumerate(pbar):            
            if i == 0 and epoch == 0:
                for x in [I.parse_input() + " # " + I.parse_output() for I in batch]:
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
        epoch: int,
        solver: MathSolver,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        val_acc  = []

        test_dataset = DefaultDataset(test_data)
        
        os.makedirs("tmp", exist_ok=True)
        f = open("tmp/{}_save_output_{}.txt".format(self.cfg.dataset_name, epoch), "w")

        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            output_OpSeq_list = solver.generate(input_text, nums, const_nums)
            target_OpSeq_list = obj["OpSeq_list"]

            try:
                output_value = compute_OpSeq_list(output_OpSeq_list, nums, const_nums, self.cfg.max_nums_size)
                target_value = compute_OpSeq_list(target_OpSeq_list, nums, const_nums, self.cfg.max_nums_size)
            except SyntaxError:
                output_value = None
                target_value = None
            eps = 1e-5

            # TEST USE
            # if i < 20:
            #     logger.info("i: {}".format(i))
            #     logger.info("input_text: {}".format(input_text))
            #     logger.info("output_Op_list: {}".format(output_OpSeq_list))
            #     logger.info("target_Op_list: {}".format(target_OpSeq_list))
            #     logger.info("output_value: {}".format(output_value))
            #     logger.info("target_value: {}".format(target_value))

            if (output_value is not None and target_value is not None and abs(output_value - target_value) < eps):
                val_acc.append(1)
            else:
                val_acc.append(0)
            
            op_seq_list0 = [", ".join(x.expr_toks) for x in output_OpSeq_list]
            op_seq_list1 = [", ".join(x.expr_toks) for x in target_OpSeq_list]
            f.write("test-id: {}\n".format(i))
            f.write("input={}\n".format(input_text))
            f.write("nums={}\n".format(nums))
            f.write("output={}\n".format(op_seq_list0))
            f.write("target={}\n".format(op_seq_list1))
            f.write("correct={}\n".format("True" if val_acc[-1] == 1 else "False"))

        f.close()

        for name, acc in zip(["val_acc"], [val_acc]):
            msg = "epoch: {} {}: {}".format(epoch, name, sum(acc) / len(acc))
            logger.info(msg)
