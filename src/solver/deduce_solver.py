import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from transformers.models.bert import BertModel, BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union
from tqdm import tqdm

from utils import Tok, Op, build_Op_list, compute_Op_list
from cfg import Config


class DeducePreprocesser:
    
    def __init__(self, ops: List[Tok], constant_nums: List[Tok], max_nums_size: int) -> None:
        self.ops_size = len(ops)
        self.max_nums_size = max_nums_size
        self.constant_nums_size = len(constant_nums)
        self.ops = ops
        self.constant_nums = constant_nums
        self.op2id = {
            w: i for i, w in enumerate(ops)
        }

    def batch_process(self, batch_seg_expr: List[List[Tok]], batch_nums: List[List[int]]):
        batch_Op_list = []
        for seg_expr, nums in zip(batch_seg_expr, batch_nums):
            batch_Op_list.append(build_Op_list(seg_expr, nums))
        
        max_Op_length = max(len(Ops) for Ops in batch_Op_list) + 1
        for Op_list in batch_Op_list:
            Op_list.extend([None] * (max_Op_length - len(Op_list)))
        
        return batch_Op_list


class Attention(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.inf = 1e12
        self.W = nn.Linear(2 * hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, m: Tensor, mask: Tensor) -> Tensor:
        """
            input:
                x: [N, D] | [N, L, D]
                m: [N, L', D]
                m_mask: [N, L']
        """
        dim_equal_two = (x.dim() == 2)
        if dim_equal_two:
            x = x.unsqueeze(dim=1)

        l0 = x.shape[1]
        l1 = m.shape[1]

        x_c = x.unsqueeze(dim=2).repeat([1, 1, l1, 1])  # x_copy [N, L, L', D]
        m_c = m.unsqueeze(dim=1).repeat([1, l0, 1, 1])  # m_copy [N, L, L', D]

        w: Tensor = self.V(torch.tanh(self.W(torch.cat((x_c, m_c), dim=-1)))).squeeze(dim=-1)  # [N, L, L']
        w = w.masked_fill((~(mask.bool())).unsqueeze(dim=1), -self.inf)

        w = torch.softmax(w, dim=-1)  # [N, L, L']
        w = w.unsqueeze(dim=-1)  # [N, L, L', 1]

        v = torch.sum(w * m_c, dim=-2)  # value [N, L, D]
        if dim_equal_two:
            v = v.squeeze(dim=1)
        return v


class DeduceDecoder(nn.Module):

    def __init__(self, hidden_dim: int, ops_size: int, constant_nums_size: int, max_nums_size: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ops_size = ops_size
        self.constant_nums_size = constant_nums_size
        self.max_nums_size = max_nums_size
        self.inf = 1e5
        
        self.constant_nums_embedding = nn.parameter\
            .Parameter(torch.randn(self.constant_nums_size, hidden_dim))

        self.bos_embedding = nn.parameter\
            .Parameter(torch.randn(hidden_dim))
        
        self.ops_Modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(self.ops_size)
        ])

        self.attn = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cor_attn = Attention(hidden_dim)

        self.ffn_switch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def cor(
        self,
        x: Tensor,  # [N, D]
        memory: Tensor,       # [N, L, D]
        memory_mask: Tensor,  # [N, L]
    ):
        return x + self.cor_attn(x, memory, memory_mask)

    def forward(
        self,
        input_embedding: Tensor,  # [N, D]
        past_value: Tensor,  # [N, D],
        encoder_outputs: Tensor,  # [N, L', D]
        nums_embedding: Tensor,  # [N, MAX_NUM_SIZE, D],
        attention_mask: Tensor,  # [N, L'],
        nums_mask: Tensor,  # [N, MAX_NUM_SIZE]
        deduce_mask: Tensor,  # [N, (MAX_NUM_SIZE ** 2) * OPS_SIZE]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N = input_embedding.shape[0]
        input_embedding = input_embedding.unsqueeze(1)
        past_value = past_value.unsqueeze(dim=0)
        query, hn = self.gru(input_embedding, past_value)  # qs [N, 1, D], hn [1, N, D]
        ctx = self.attn(query, encoder_outputs, attention_mask)  # context0 [N, 1, D]

        act_embedding = (query + ctx).squeeze(1)  # feature [N, D]

        stop_prob = torch.sigmoid(self.ffn_switch(act_embedding).squeeze(-1))  # [N]
        
        var0 = nums_embedding.unsqueeze(2).repeat(1, 1, self.max_nums_size, 1)\
            .reshape(N, self.max_nums_size ** 2, self.hidden_dim)  # [N, MAX_NUM_SIZE ** 2, D]
        var1 = nums_embedding.unsqueeze(1).repeat(1, self.max_nums_size, 1, 1)\
            .reshape(N, self.max_nums_size ** 2, self.hidden_dim)  # [N, MAX_NUM_SIZE ** 2, D]
        
        arg_embedding = torch.cat((var0, var1, var0 * var1), dim=-1)  # [N, (MAX_NUM_SIZE ** 2), 4 * D]

        ops_arg_embedding_list = []
        for module in self.ops_Modules:
            ops_arg_embedding_list.append(module(arg_embedding))
        
        expr_embedding = torch.cat(ops_arg_embedding_list, dim=1)  # [N, OPS_SIZE * (MAX_NUM_SIZE ** 2), D]
        
        logits = torch.sum(expr_embedding * act_embedding.unsqueeze(1), dim=-1)
        logits = logits.masked_fill((~deduce_mask), -self.inf)

        return stop_prob, logits, expr_embedding, hn.squeeze(0)


class BertEncoder(nn.Module):

    def __init__(self, bert_name: str) -> None:
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained(bert_name)
    
    def forward(
        self,
        input_dict: Dict[str, Tensor]
    ) -> Tensor:
        encoder_outputs = self.bert(**input_dict).last_hidden_state  # [N, L, D]
        return encoder_outputs


class DeduceSolver:
    
    def __init__(
        self,
        cfg_dict: Dict[AnyStr, Any],
        op_words: List[str],
        constant_nums: List[str],
    ) -> None:
        self.cfg = Config(**cfg_dict)

        self.enc_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.cfg.model_name)
        self.encoder = BertEncoder(self.cfg.model_name)

        self.deduce_preprocesser = DeducePreprocesser(op_words, constant_nums, self.cfg.max_nums_size)
        self.deducer = DeduceDecoder(
            self.cfg.model_hidden_size, 
            self.deduce_preprocesser.ops_size,
            self.deduce_preprocesser.constant_nums_size,
            self.deduce_preprocesser.max_nums_size
        )
        
        self.num_tokens_ids: List[int] = None
        self.dense_index = None

        self.update_voc()
        self.to_device(self.cfg.device)

    def save_model(self, dir_path: str, suffix: str = "") -> None:
        enc_path = os.path.join(dir_path, f"deduce_enc_{suffix}.pth")
        torch.save(self.encoder.state_dict(), enc_path)
        dec_path = os.path.join(dir_path, f"deduce_dec_{suffix}.pth")
        torch.save(self.deducer.state_dict(), dec_path)
         
    def load_model(self, dir_path: str, suffix: str = "") -> None:
        enc_path = os.path.join(dir_path, f"deduce_enc_{suffix}.pth")
        self.encoder.load_state_dict(torch.load(enc_path))
        dec_path = os.path.join(dir_path, f"deduce_dec_{suffix}.pth")
        self.deducer.load_state_dict(torch.load(dec_path))

    def to_device(self, device: str) -> None:
        self.cfg.device = device
        self.encoder.to(device)
        self.deducer.to(device)
    
    def train(self) -> None:
        self.encoder.train()
        self.deducer.train()
    
    def eval(self) -> None:
        self.encoder.eval()
        self.deducer.eval()

    def update_voc(self) -> None:
        new_tokens = ['[num]'] \
            + [f'[num{n}]' for n in range(self.cfg.max_nums_size)] \
            + ['[int]', '[float]', '[frac]', '[perc]'] \
            + [f'[rk{n}]' for n in range(self.cfg.max_nums_size)]
        self.enc_tokenizer.add_tokens(new_tokens)
        self.encoder.bert.resize_token_embeddings(len(self.enc_tokenizer))

        self.num_tokens_ids = self.enc_tokenizer.convert_tokens_to_ids(
            [f'[num{n}]' for n in range(self.cfg.max_nums_size)]
        )

    def build_nums_mask(self, nums_size: List[int]) -> Tensor:
        N = len(nums_size)
        nums_mask = torch.zeros(N, self.deducer.max_nums_size, dtype=torch.bool, device=self.cfg.device)
        for i in range(N):
            nums_mask[i, :nums_size[i]] = True
        return nums_mask

    def build_deduce_mask(self, nums_size: List[int]) -> Tensor:
        N = len(nums_size)
        deduce_mask = torch.zeros(N, (self.deducer.max_nums_size ** 2) * self.deducer.ops_size, 
                                  dtype=torch.bool, device=self.cfg.device)
        for i in range(N):
            mask = torch.zeros(self.deducer.max_nums_size, dtype=torch.bool, device=self.cfg.device)
            mask[:nums_size[i]] = True
            mask = (mask.unsqueeze(1) & mask.unsqueeze(0)).flatten().repeat(self.deducer.ops_size)
            deduce_mask[i, :].copy_(mask)
        
        # add priori-knowledge
        # ...

        return deduce_mask

    def convert_op2index(self, op: Op):
        op_id = self.deduce_preprocesser.op2id[op.op]
        index = op_id * (self.deducer.max_nums_size ** 2) + (op.arg1 * self.deducer.max_nums_size + op.arg2)
        return index

    def convert_index2op(self, index: int):
        op_id = index // (self.deducer.max_nums_size ** 2)
        index = index % (self.deducer.max_nums_size ** 2)
        op = self.deduce_preprocesser.ops[op_id]
        arg1 = index // self.deducer.max_nums_size
        arg2 = index % self.deducer.max_nums_size
        return arg1, arg2, op
    
    def prepare_input(
        self,
        batch_tokens: List[List[Tok]]
    ) -> Tuple[Dict[str, Tensor], List[List[int]]]:
        batch_text = ["".join(tokens) for tokens in batch_tokens]
        input_dict = self.enc_tokenizer.batch_encode_plus(batch_text, return_tensors="pt", padding=True)
        nums_ids = []
        for ids in input_dict.input_ids:
            nums_ids.append([i for i, v in enumerate(ids) if v.item() in self.num_tokens_ids])
        return input_dict.to(self.cfg.device), nums_ids
        
    def encode(
        self, 
        input_dict: Dict[str, Tensor], 
        batch_num_ids: List[List[int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        N = len(batch_num_ids)
        encoder_outputs: Tensor = self.encoder(input_dict)
        nums_embedding = torch.zeros(N, self.cfg.max_nums_size, self.cfg.model_hidden_size, dtype=torch.float, device=encoder_outputs.device)
        for i in range(N):
            n = len(batch_num_ids[i])
            nums_embedding[i, :n, :].copy_(encoder_outputs[i, batch_num_ids[i], :])
            nums_embedding[i, n:n + self.deducer.constant_nums_size, :].copy_(self.deducer.constant_nums_embedding)
        text_embedding = encoder_outputs[:, 0, :].clone()
        return text_embedding, encoder_outputs, nums_embedding

    def prepare_output(
        self, 
        batch_seg_expr: List[List[Tok]],
        batch_nums: List[List[str]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_Op_list = self.deduce_preprocesser.batch_process(batch_seg_expr, batch_nums)
        return batch_Op_list

    def decode(
        self,
        input_embedding: Tensor,  # [N, D]
        past_value: Tensor,  # [N, D],
        encoder_outputs: Tensor,  # [N, L', D]
        nums_embedding: Tensor,  # [N, MAX_NUM_SIZE, D],
        attention_mask: Tensor,  # [N, L'],
        nums_mask: Tensor,  # [N, MAX_NUM_SIZE]
        deduce_mask: Tensor,  # [N, (MAX_NUM_SIZE ** 2) * OPS_SIZE]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.deducer(
            input_embedding,
            past_value,
            encoder_outputs,
            nums_embedding,
            attention_mask,
            nums_mask,
            deduce_mask,
        )

    @torch.no_grad()
    def generate(self, tokens: List[Tok], nums: List[str], max_length: int = 35) -> float:
        self.eval()

        input_dict, nums_ids = self.prepare_input([tokens])
        text_embedding, encoder_outputs, nums_embedding = self.encode(input_dict, nums_ids)

        input_embedding = self.deducer.bos_embedding.unsqueeze(0)
        past_value = text_embedding
        attention_mask = input_dict["attention_mask"]

        nums_size = [len(nums)]
        
        Op_list = []
        deduce_step = 0

        max_length = max_length - len(nums)
        
        while deduce_step < max_length:
            nums_mask = self.build_nums_mask(nums_size)
            deduce_mask = self.build_deduce_mask(nums_size)
            
            stop_prob, logits, expr_embedding, hn = self.decode(  # logits: [1, (MAX_NUMS_SIZE ** 2) * OPS_SIZE]
                input_embedding,
                past_value,
                encoder_outputs,
                nums_embedding,
                attention_mask,
                nums_mask,
                deduce_mask,
            )
            
            if stop_prob.item() > 0.5:
                break
            
            target_op = torch.argmax(logits, dim=1).item()
            arg1, arg2, op = self.convert_index2op(target_op)
            arg0 = nums_size[0]
            Op_list.append(Op(arg0, arg1, arg2, op))
            
            past_value = hn
            nums_embedding[0, arg0, :].copy_(expr_embedding[0, target_op, :])
            input_embedding = self.deducer.cor(
                nums_embedding[0, arg0, :].clone().unsqueeze(0),
                encoder_outputs,
                attention_mask
            )
            
            nums_size[0] += 1
            deduce_step += 1

        return compute_Op_list(Op_list, nums, self.deducer.max_nums_size)
