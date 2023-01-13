import json
import os
import math
import random
from solver import MathSolverRE
from scheduler import GradualWarmupScheduler
from math_utils import MathDataset, compute_Expr_list
from cfg import MathConfig
from math_utils import MathDataInstance
from transformers import get_linear_schedule_with_warmup

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


class MathTrainerRE:

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
    
    def augment_data(self):
        logger.info("data augumentation[shuffle,mix]")
        shuffle_train_dataset = self.add_shuffle(self.raw_dataset["raw_train"])
        mix_train_dataset = self.add_mix(self.raw_dataset["raw_train"])
        self.raw_dataset["train"] = deepcopy(self.raw_dataset["raw_train"])
        self.raw_dataset["train"].extend(shuffle_train_dataset)
        self.raw_dataset["train"].extend(mix_train_dataset)

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

    def add_shuffle(self, data: List[Dict], num: int = 1000):
        num = min(num, len(data))
        indices = np.random.choice(range(len(data)), num, replace=False)
        data = [data[i] for i in indices]
        new_data = []
        for obj in data:
            obj = deepcopy(obj)
            text: str = obj["seg_text"]
            seg_text = text.split(".")
            seg_text, q_text = seg_text[:-1], seg_text[-1:]
            random.shuffle(seg_text)
            seg_text = ".".join(seg_text + q_text)
            obj["id"] = -1
            obj["seg_text"] = seg_text
            new_data.append(obj)
        return new_data

    def add_mix(self, data: List[Dict], num: int = 1000):
        num = min(num, len(data))
        indices1 = np.random.choice(range(len(data)), num, replace=False)
        indices2 = np.random.choice(range(len(data)), num, replace=False)
        new_data = []
        for i, j in zip(indices1, indices2):
            if i == j:
                continue
            obj1 = data[i]
            obj2 = data[j]
            nums_size1 = len(obj1["nums"])
            nums_size2 = len(obj2["nums"])
            text1: str = obj1["seg_text"]
            text2: str = obj2["seg_text"]
            for i in range(nums_size2 - 1, -1, -1):
                text2 = text2.replace(f"[num{i}]", f"[num{i+nums_size1}]")
            seg_text1 = text1.split(".")
            seg_text2 = text2.split(".")
            ctx = seg_text1[:-1] + seg_text2[:-1]
            q_text = seg_text1[-1:]
            random.shuffle(ctx)
            text = ".".join(ctx + q_text)
            nums = obj1["nums"] + obj2["nums"]
            obj = {
                "id": -2,
                "sample_id": obj1["sample_id"],
                "seg_text": text,
                "nums": nums,
                "const_nums": obj1["const_nums"],
                "Expr_list": obj1["Expr_list"]
            }
            new_data.append(obj)
        return new_data

    def convert_dataset(self, dataset: List[Dict[AnyStr, Any]]) -> List[MathDataInstance]:
        new_dataset = []
        for obj in dataset:
            question = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            expr_list = obj["Expr_list"]
            for i in range(len(expr_list)):
                new_dataset.append(MathDataInstance(
                    question=question,
                    nums=nums,
                    const_nums=const_nums,
                    expr_list=expr_list[:i],
                    target=[expr_list[i]],
                    id=obj["sample_id"],
                    end=(i + 1 == len(expr_list))
                ))
        return new_dataset

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: MathSolverRE):
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

        print("num_training_steps = {}".format(self.cfg.num_epochs * len(loader)))

        scheduler = get_linear_schedule_with_warmup(
            optim, 
            num_warmup_steps=0,
            num_training_steps=self.cfg.num_epochs * len(loader)
        )
        
        for epoch in range(self.cfg.num_epochs):
            if "svamp" in self.cfg.dataset_name and self.cfg.use_data_aug:
                self.augment_data()
                self.train_dataset = self.convert_dataset(self.raw_dataset["train"])
                dataset.data = self.train_dataset
                
            self.train_one_epoch(epoch, solver, optim, scheduler, loader)
            
            if (epoch > self.cfg.num_epochs // 2 and epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5) or (epoch == 0):
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
        solver: MathSolverRE,
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
    def evaluate(
        self,
        dataset_type: str,
        epoch: int,
        solver: MathSolverRE,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        Acc  = []

        test_dataset = MathDataset(test_data)
                
        hist_count   = [0 for _ in range(20)]
        hist_correct = [0 for _ in range(20)]
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            output_Expr_list = solver.beam_search(input_text, nums, const_nums, beam_size=self.cfg.beam_size)
            target_Expr_list = obj["Expr_list"]

            length = len(target_Expr_list)
            
            try:
                output_value = compute_Expr_list(output_Expr_list, nums, const_nums, self.cfg.quant_size)
                target_value = compute_Expr_list(target_Expr_list, nums, const_nums, self.cfg.quant_size)
            except SyntaxError:
                output_value = None
                target_value = None
            eps = 1e-5
            
            if (output_value is not None and target_value is not None and abs(output_value - target_value) < eps):
                Acc.append(1)
                hist_count[length]   += 1
                hist_correct[length] += 1
            else:
                Acc.append(0)
                hist_count[length]   += 1

        if self.cfg.save_result:
            os.makedirs("../cache/mwp", exist_ok=True)
            hist_acc = [hist_correct[i] / (hist_count[i] + 1e-5) for i in range(20)]
            f = open("../cache/mwp/{}_{}_{}_re.json".format(self.cfg.dataset_name, dataset_type, epoch), "w")
            f.write(json.dumps(hist_acc))
            f.close()

        answer_mAcc = sum(Acc) / len(Acc)
        msg = "epoch: {} answer-mAcc: {}".format(epoch, answer_mAcc)
        logger.info(msg)

        return answer_mAcc


    @torch.no_grad()
    def evaluate_all_beams(
        self,
        dataset_type: str,
        epoch: int,
        solver: MathSolverRE,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        Acc  = []

        test_dataset = MathDataset(test_data)
                
        hist_count   = [0 for _ in range(20)]
        hist_correct = [0 for _ in range(20)]
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            nums = obj["nums"]
            const_nums = obj["const_nums"]
            
            output_Expr_lists = solver.beam_search(input_text, nums, const_nums, beam_size=self.cfg.beam_size, return_all=True)
            target_Expr_list = obj["Expr_list"]

            length = len(target_Expr_list)
            target_value = compute_Expr_list(target_Expr_list, nums, const_nums, self.cfg.quant_size)            
            
            if output_Expr_lists is None:
                Acc.append(0)
                hist_count[length]   += 1
            else:
                o_values = []
                o_lengths = []
                b_scores = []
                for output_Expr_list, score in output_Expr_lists:
                    try:
                        o_value = compute_Expr_list(output_Expr_list, nums, const_nums, self.cfg.quant_size)
                    except SyntaxError:
                        o_value = None
                    if o_value is not None:
                        o_lengths.append(len(output_Expr_list))
                        b_scores.append(score)
                        o_values.append(o_value)

                eps = 1e-5
                def eq(x, y):
                    return abs(x - y) < eps
                
                if len(o_values) == 2 and not eq(o_values[0], target_value) and eq(o_values[1], target_value):
                    logger.info("o_values={}, b_scores={}, o_lengths={}.".format(o_values, b_scores, o_lengths))
                
                p_value = None
                if len(o_values) > 0:
                    p_value = o_values[0]
                
                if p_value is not None and eq(p_value, target_value):
                    Acc.append(1)
                    hist_count[length]   += 1
                    hist_correct[length] += 1
                else:
                    Acc.append(0)
                    hist_count[length]   += 1

        if self.cfg.save_result:
            os.makedirs("../cache/mwp", exist_ok=True)
            hist_acc = [hist_correct[i] / (hist_count[i] + 1e-5) for i in range(20)]
            f = open("../cache/mwp/{}_{}_{}_re.json".format(self.cfg.dataset_name, dataset_type, epoch), "w")
            f.write(json.dumps(hist_acc))
            f.close()

        answer_mAcc = sum(Acc) / len(Acc)
        msg = "epoch: {} answer-mAcc: {}".format(epoch, answer_mAcc)
        logger.info(msg)

        return answer_mAcc
