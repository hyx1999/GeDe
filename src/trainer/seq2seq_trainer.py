from solver import Seq2seqSolver
from scheduler import GradualWarmupScheduler
from utils import DefaultDataset, compute_expr
from cfg import Config

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


class Seq2seqTrainer:

    def __init__(
        self, 
        cfg_dict: Dict[AnyStr, Any],
        train_dataset: List[Dict[AnyStr, Any]],
        test_dataset: List[Dict[AnyStr, Any]]
    ) -> None:
        self.cfg = Config(**cfg_dict)
        self.train_dataset = deepcopy(train_dataset)
        self.test_dataset = deepcopy(test_dataset)

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch   

    def train(self, solver: Seq2seqSolver):
        solver.to_device(self.cfg.device)

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
            
            if epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
                logger.info("[evaluate test-data]")
                # self.evaluate(epoch, solver, self.test_dataset)
                self.evaluate(epoch, solver, self.train_dataset[:20])

    def train_one_epoch(
        self,
        epoch: int,
        solver: Seq2seqSolver,
        optim: Union[Adam, AdamW],
        loader: DataLoader
    ) -> None:
        solver.train()

        pbar = tqdm(loader, desc="baseline-train", total=len(loader))

        loss_total = 0
        for batch in pbar:            
            batch_input = [obj["seg_text"] for obj in batch]
            batch_output = [obj["seg_expr"] for obj in batch]

            input_dict, nums_ids = solver.prepare_input(batch_input)
            decoder_input_ids, target_ids = solver.prepare_output(batch_output)

            text_embedding, encoder_outputs, W_nums, output_mask = solver.encode(input_dict, nums_ids)
            decoder_output_logits, _ = solver.decode(
                decoder_input_ids=decoder_input_ids,
                past_value=text_embedding,
                W_nums=W_nums,
                output_mask=output_mask,
                encoder_outputs=encoder_outputs,
                attention_mask=input_dict["attention_mask"]
            )            
            decoder_output_logits = decoder_output_logits.transpose(1, 2)

            loss = F.cross_entropy(decoder_output_logits, target_ids, ignore_index=-100)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_total += loss.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}")

# --------------------------------------------- evaluate ------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        epoch: int,
        solver: Seq2seqSolver,
        test_data: List[Dict]
    ) -> float:
        solver.eval()
        
        expr_acc = []
        val_acc  = []

        test_dataset = DefaultDataset(test_data)
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            target_text = "".join(obj["seg_expr"])
            nums = obj["nums"]

            output_text = solver.generate(input_text)

            if i < 20:
                logger.info("i: {}".format(i))
                logger.info("input_text: {}".format(input_text))
                logger.info("output_text: {}".format(output_text))
                logger.info("target_text: {}".format(target_text))


            if output_text == target_text:
                expr_acc.append(1)
            else:
                expr_acc.append(0)

            eps = 1e-5
            v0 = compute_expr(output_text, nums)
            v1 = compute_expr(target_text, nums)
            if (v0 is not None and v1 is not None and abs(v0 - v1) < eps) or output_text == target_text:
                val_acc.append(1)
            else:
                val_acc.append(0)
    
        for name, acc in zip(["expr_acc", "val_acc"], [expr_acc, val_acc]):
            msg = "epoch: {} {}: {}".format(epoch, name, sum(acc) / len(acc))
            print(msg)
            logger.info(msg)
