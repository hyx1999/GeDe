from solver import DeduceSolver
from scheduler import GradualWarmupScheduler
from utils import Tok, Op, DefaultDataset, compute_Op_list
from cfg import Config

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import random
from loguru import logger
from typing import Dict, Any, AnyStr, List, Optional, Tuple, Union
from copy import deepcopy
from tqdm import tqdm


class DeduceTrainer:

    def __init__(
        self, 
        cfg_dict: Dict[AnyStr, Any],
        train_dataset: List[Dict[AnyStr, Any]],
        test_dataset: List[Dict[AnyStr, Any]],
    ) -> None:
        self.cfg = Config(**cfg_dict)
        self.train_dataset = deepcopy(train_dataset)
        self.test_dataset = deepcopy(test_dataset)

    def collate_fn(
        self, 
        batch: List[Dict[AnyStr, Any]]
    ) -> List[Dict[AnyStr, Any]]:
        return batch

    def train(self, solver: DeduceSolver):
        solver.to_device(self.cfg.device)
                
        optim = AdamW(
            [
                {'params': solver.encoder.parameters(), 'lr': 5e-5},
                {'params': solver.deducer.parameters(), 'lr': 5e-5}, 
            ],
            weight_decay=1e-4
        )

        scheduler_steplr = StepLR(optim, step_size=self.cfg.scheduler_step_size, gamma=0.5)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_steplr)
                        
        train_dataset = DefaultDataset(self.train_dataset)
        loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        for epoch in range(self.cfg.num_epochs):
            self.train_one_epoch(epoch, solver, optim, loader)
            scheduler_warmup.step()
            
            if epoch % 5 == 0 or epoch > self.cfg.num_epochs - 5:
                logger.info("[evaluate test-data]")
                self.evaluate(epoch, solver, self.test_dataset)
        
    def train_one_epoch(
        self,
        epoch: int,
        solver: DeduceSolver,
        optim: Union[Adam, AdamW],
        loader: DataLoader
    ) -> None:
        solver.train()

        pbar = tqdm(loader, desc="train deduce model", total=len(loader))

        loss_total = 0
        for batch in pbar:
            batch_input = [obj["seg_text"] for obj in batch]
            batch_output = [obj["seg_expr"] for obj in batch]
            batch_nums = [obj["nums"] for obj in batch]

            input_dict, nums_ids = solver.prepare_input(batch_input)
            batch_Op_list = solver.prepare_output(batch_output, batch_nums)

            text_embedding, encoder_outputs, nums_embedding = solver.encode(input_dict, nums_ids)

            N = len(batch)

            input_embedding = solver.deducer.bos_embedding.unsqueeze(0).repeat(N, 1)
            past_value = text_embedding
            attention_mask = input_dict["attention_mask"]

            nums_size = [len(batch_nums[i]) for i in range(N)]
            
            losses = []

            finish_mask = [False] * N
            Op_index = 0
            
            while sum(finish_mask) < N:
                
                nums_mask = solver.build_nums_mask(nums_size)
                deduce_mask = solver.build_deduce_mask(nums_size)
                
                stop_prob, logits, expr_embedding, hn = solver.decode(
                    input_embedding,
                    past_value,
                    encoder_outputs,
                    nums_embedding,
                    attention_mask,
                    nums_mask,
                    deduce_mask,
                )
                
                loss_mask = ~torch.tensor(finish_mask, dtype=torch.bool, device=self.cfg.device)
                
                target_switch = torch.tensor([1.0 if Op_list[Op_index] is None else 0.0 for Op_list in batch_Op_list], 
                                            dtype=torch.float, device=self.cfg.device)
                weight_switch = torch.tensor([(Op_index + 1.0) if Op_list[Op_index] is None else 1.0 for Op_list in batch_Op_list], 
                                            dtype=torch.float, device=self.cfg.device)
                
                target_op = torch.tensor(
                    [solver.convert_op2index(Op_list[Op_index]) if Op_list[Op_index] is not None else -1 for Op_list in batch_Op_list],
                    dtype=torch.long,
                    device=self.cfg.device
                )
                                
                loss_switch = F.binary_cross_entropy(stop_prob[loss_mask], target_switch[loss_mask], weight=weight_switch[loss_mask])
                loss_op = F.cross_entropy(logits, target_op, ignore_index=-1)
                
                loss_one_step = loss_switch + loss_op
                losses.append(loss_one_step)

                target_op_cpu = target_op.detach().cpu().numpy()
                                                
                # for i in range(N):
                #     if target_op_cpu[i] != -1:
                #         logger.info("logits[target]={}".format(logits[i, target_op_cpu[i]]))
                
                # print("loss_step: {}".format(loss_switch.item()))
                # print("loss_op: {}".format(loss_op.item()))
                
                past_value = hn
                next_input_embedding = []
                for i in range(N):
                    if batch_Op_list[i][Op_index] is not None:
                        j = target_op_cpu[i]
                        nums_embedding[i, nums_size[i], :].copy_(expr_embedding[i, j, :])
                        nums_size[i] += 1
                    next_input_embedding.append(nums_embedding[i, nums_size[i], :].clone())

                input_embedding = solver.deducer.cor(
                    torch.stack(next_input_embedding, dim=0),
                    encoder_outputs,
                    attention_mask
                )
                
                finish_mask = [True if Op_list[Op_index] is None else False for Op_list in batch_Op_list]
                Op_index += 1
            
            loss = torch.stack(losses).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            loss_total += loss.item()
            pbar.set_postfix_str("loss: {:.5f}".format(loss.item()))

        loss_ave = loss_total / len(loader)
        logger.info(f"epoch: {epoch}, loss-ave: {loss_ave}")

    @torch.no_grad()
    def evaluate(
        self,
        epoch: int,
        solver: DeduceSolver,
        test_dataset: List[Dict]
    ) -> float:
        solver.eval()
        
        val_acc  = []
        
        for i in tqdm(range(len(test_dataset)), desc="evaluate", total=len(test_dataset)):
            obj = test_dataset[i]
            input_text = "".join(obj["seg_text"])
            target_Op_list = obj["Op_list"]
            nums = obj["nums"]

            output_value = solver.generate(input_text, nums)
            target_value = compute_Op_list(target_Op_list, nums, solver.deducer.max_nums_size)
            eps = 1e-5

            if (output_value is not None and target_value is not None and abs(output_value - target_value) < eps):
                val_acc.append(1)
            else:
                val_acc.append(0)
    
        msg = "epoch: {} value accuracy: {:.3f}".format(epoch, sum(val_acc) / len(val_acc))
        print(msg)
        logger.info(msg)
