import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.bert import BertModel, BertTokenizer, BertConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union
from tqdm import tqdm

from loguru import logger
from utils import Op, OpSeq, Tok, OpSeqDataInstance
from cfg import MathConfig

class Attention(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, 
        input_emb: Tensor, 
        memory: Tensor, 
        cross_attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
            input:
                x: [N, D] | [N, L, D]
                m: [N, L', D]
                cross_attn_mask: [N, L, L']
        """
        dim2 = (input_emb.dim() == 2)
        if dim2:
            input_emb = input_emb.unsqueeze(dim=1)

        input_length = input_emb.shape[1]
        memory_length = memory.shape[1]

        var0 = input_emb.unsqueeze(dim=2).repeat([1, 1, memory_length, 1])  # x_copy [N, L, L', D]
        var1 = memory.unsqueeze(dim=1).repeat([1, input_length, 1, 1])  # m_copy [N, L, L', D]

        weight: Tensor = self.fc1(torch.tanh(self.fc0(torch.cat((var0, var1), dim=-1)))).squeeze(dim=-1)  # [N, L, L']
        if cross_attn_mask is not None:
            weight = weight.masked_fill((~(cross_attn_mask.bool())), torch.finfo(torch.float).min)

        weight = torch.softmax(weight, dim=-1)  # [N, L, L']
        weight = weight.unsqueeze(dim=-1)  # [N, L, L', 1]

        output = torch.sum(weight * var1, dim=-2)  # value [N, L, D]
        if dim2:
            output = output.squeeze(dim=1)
        return output


class SeqDecoder(nn.Module):

    def __init__(self, hidden_dim: int, ext_size: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.attn = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ext_emb = nn.parameter.Parameter(
            torch.randn(ext_size, hidden_dim)
        )
        self.ext_emb_out = nn.parameter.Parameter(
            torch.randn(ext_size, hidden_dim)
        )


    def embedding(self, input_ids: Tensor, voc_emb: Tensor) -> Tensor:
        input_ids = input_ids.unsqueeze(dim=-1).repeat([1, 1, self.hidden_dim])  # [N, L, D]
        input_emb = torch.gather(voc_emb, dim=1, index=input_ids)
        return input_emb
    
    def forward(
        self,
        decoder_input_ids: Tensor,  # [N] or [N, L]
        past_value: Tensor,  # [N, D],
        num_emb: Tensor,  # [N, num_size, D],
        voc_mask: Tensor,  # [N, voc_size]
        memory: Tensor,  # [N, L', D]
        cross_attn_mask: Optional[Tensor] = None,  # [N, L']
    ) -> Tensor:
        dim1 = (decoder_input_ids.dim() == 1)
        if dim1:
            decoder_input_ids = decoder_input_ids.unsqueeze(dim=-1)  # [N] -> [N, L]

        bs = decoder_input_ids.shape[0]
        ext_emb = self.ext_emb.unsqueeze(0).repeat(bs, 1, 1)
        ext_emb_out = self.ext_emb_out.unsqueeze(0).repeat(bs, 1, 1)

        voc_emb = torch.cat((ext_emb, num_emb), dim=1)  # [N, V, D]
        voc_emb_out = torch.cat((ext_emb_out, num_emb), dim=1)  # [N, V, D]

        decoder_inputs = self.embedding(decoder_input_ids, voc_emb)
        qs, hn = self.gru(decoder_inputs, past_value.unsqueeze(dim=0))  # queries [N, L, D], hn [1, N, D]
        cs = self.attn(qs, memory, cross_attn_mask)  # contexts [N, L, D]
        feat = torch.tanh(self.fc(torch.cat((qs, cs), dim=-1)))  # feature [N, L, D]

        # logits = torch.bmm(feat, voc_emb.transpose(1, 2))  # [N, L, voc_size]
        logits = torch.bmm(feat, voc_emb_out.transpose(1, 2))  # [N, L, voc_size]
        logits = logits.masked_fill((~voc_mask).unsqueeze(dim=1), torch.finfo(torch.float).min)
        if dim1:
            logits = logits.squeeze(dim=1)  # [N, voc_size]
        return logits, hn.squeeze(dim=0)


class BertEncoder(nn.Module):

    def __init__(self, bert_name: str) -> None:
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained(
            bert_name,
            cache_dir="../data/cache"
        )
    
    def forward(
        self,
        input_dict: Dict[str, Tensor]
    ) -> Tensor:
        encoder_outputs = self.bert(**input_dict).last_hidden_state  # [N, L, D]
        return encoder_outputs


class MathSolver(nn.Module):
    
    def __init__(
        self,
        cfg_dict: Dict[AnyStr, Any],
        const_nums: List[str]
    ) -> None:
        super().__init__()
        self.cfg = MathConfig(**cfg_dict)

        self.tok: BertTokenizer = BertTokenizer.from_pretrained(
            self.cfg.model_name,
            cache_dir="../data/cache"
        )
        self.update_voc(len(const_nums))

        self.encoder = BertEncoder(self.cfg.model_name)

        self.decoder = SeqDecoder(
            self.encoder.bert.config.hidden_size,
            len(self.ext_tokens) + len(self.const_nums_id)
        )

        self.encoder.bert.resize_token_embeddings(len(self.tok))

        # 21130, 120, 21131, 118, 21130, 120, 21132, 138, 10434, 8148, 140
        # print(self.tok.convert_ids_to_tokens([21130, 120, 21131, 118, 21130, 120, 21132]))
        # print(self.tok.convert_ids_to_tokens([10434, 8148, 140]))
        # exit(0)

    def save_model(self, dir_path: str, suffix: str = "") -> None:
        path = os.path.join(dir_path, f"seq2seqv2_{suffix}.pth")
        torch.save(self.state_dict(), path)
         
    def load_model(self, dir_path: str, suffix: str = "") -> None:
        path = os.path.join(dir_path, f"seq2seqv2_{suffix}.pth")
        self.load_state_dict(torch.load(path))

    def update_voc(self, const_nums_size: int):
        g_tokens  = ['[eos0]', '[eos1]', self.tok.pad_token]
        ops = ["+", "-", "*", "/", "^", "(", ")"]
        const_nums_id   = [f'[c{n}]' for n in range(const_nums_size)]
        nums_id   = [f'[num{n}]' for n in range(self.cfg.max_nums_size)]
        nums_type = ['[int]', '[float]', '[frac]', '[perc]']
        nums_rk   = [f'[rk{n}]' for n in range(self.cfg.max_nums_size)]

        new_tokens = g_tokens + ops + const_nums_id + nums_id + nums_type + nums_rk
        self.tok.add_tokens(new_tokens)
        
        self.g_tokens = g_tokens
        self.ops = ops
        self.ext_tokens = g_tokens + self.ops
        self.nums_id = nums_id
        self.const_nums_id = const_nums_id

        self.g_token_ids = self.tok.convert_tokens_to_ids(self.g_tokens)
        self.op_token_ids = self.tok.convert_tokens_to_ids(self.ops)
        self.ext_token_ids = self.tok.convert_tokens_to_ids(self.ext_tokens)
        self.const_num_token_ids = self.tok.convert_tokens_to_ids(self.const_nums_id)
        self.num_token_ids = self.tok.convert_tokens_to_ids(self.nums_id)

        self.dvoc = self.ext_tokens + self.const_nums_id + self.nums_id
        self.dvoc_token_id = {k: v for v, k in enumerate(self.dvoc)}
        self.voc2dvoc = {
            self.tok.convert_tokens_to_ids(k): v 
                for k, v in self.dvoc_token_id.items()
        }
        self.dvoc2voc = {
            k: v for v, k in self.voc2dvoc.items()
        }

        self.voc_eos0_token_id = self.tok.convert_tokens_to_ids('[eos0]')
        self.voc_eos1_token_id = self.tok.convert_tokens_to_ids('[eos1]')
        self.dvoc_eos0_token_id = self.voc2dvoc[self.voc_eos0_token_id]
        self.dvoc_eos1_token_id = self.voc2dvoc[self.voc_eos1_token_id]

    def prepare_input(
        self,
        input_text: List[str]
    ) -> Tuple[Dict[str, Tensor], List[List[int]]]:
        input_dict = self.tok(
            input_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self.cfg.device)
        seq_length = input_dict.input_ids.shape[-1]
        num_ids   = []
        attn_mask = []
        for per_ids in input_dict.input_ids:
            used = set()
            per_num_ids = []
            per_sep_ids = []
            for j, v in enumerate(per_ids):
                if v.item() in self.num_token_ids and v.item() not in used:
                    used.add(v.item())
                    per_num_ids.append(j)
                if v.item() == self.tok.sep_token_id:
                    per_sep_ids.append(j + 1)
            per_attn_mask = input_dict.attention_mask.new_zeros(seq_length, seq_length)
            p_x = 0
            for x in per_sep_ids:
                per_attn_mask[p_x:x,:x] = True
                p_x = x
            num_ids.append(per_num_ids)
            attn_mask.append(per_attn_mask)
        input_dict["attention_mask"] = torch.stack(attn_mask, dim=0)
        return input_dict, num_ids
    
    def prepare_output(
        self, 
        output_text: List[str]
    ) -> Tuple[Tensor, Tensor]:
        raw_labels: List[List[int]] = self.tok(
            output_text, 
            padding=True,
            add_special_tokens=False,
            truncation=True
        ).input_ids
        labels = torch.tensor(
            [[self.voc2dvoc[x] for x in row] for row in raw_labels],
            dtype=torch.long,
            device=self.cfg.device
        )
        shift_labels = torch.zeros_like(labels)
        shift_labels[:, 1:].copy_(labels[:, :-1])
        shift_labels[:, 0] = self.dvoc_token_id["[eos0]"]
        labels[labels == self.voc2dvoc[self.tok.pad_token_id]] = -1
        return shift_labels, labels
    
    def build_cross_attn_mask(
        self,
        input_ids: Tensor,
        labels: Tensor
    ) -> Tensor:
        bs = input_ids.shape[0]
        output_length = labels.shape[-1]
        input_length = input_ids.shape[-1]
        cross_attn_mask = torch.zeros(
            (bs, output_length, input_length),
            dtype=torch.bool
        )
        for i in range(bs):
            per_sep_ids = []
            for j, v in enumerate(input_ids[i]):
                if v.item() == self.tok.sep_token_id:
                    per_sep_ids.append(j + 1)
            per_eos_ids = []
            for j, v in enumerate(labels[i]):
                if v.item() in [self.dvoc_eos0_token_id, self.dvoc_eos1_token_id]:
                    per_eos_ids.append(j + 1)
            p_x = 0
            for x, y in zip(per_eos_ids, per_sep_ids):
                cross_attn_mask[i, p_x:x, :y] = True
                p_x = x
        return cross_attn_mask.to(self.cfg.device)
    
    def encode(
        self, 
        input_dict: Dict[str, Tensor], 
        num_ids: List[List[int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ext_size = len(self.ext_tokens) + len(self.const_nums_id)
        num_size = len(self.nums_id)
        bs = len(num_ids)
        memory: Tensor = self.encoder(input_dict)
        num_emb  = torch.zeros(
            bs, num_size, self.encoder.bert.config.hidden_size, dtype=torch.float, device=memory.device)
        voc_mask = torch.zeros(
            bs, ext_size + num_size, dtype=torch.bool, device=memory.device)
        for i in range(bs):
            n = len(num_ids[i])
            num_emb[i, :n, :].copy_(memory[i, num_ids[i], :])
            voc_mask[i, :ext_size + n] = True
        input_emb = memory[:, 0, :].clone()
        return input_emb, memory, num_emb, voc_mask

    def decode(
        self,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.decoder(*args, **kwargs)
    
    def forward(self, batch: List[OpSeqDataInstance]) -> Tensor:
        input_dict, num_ids = self.prepare_input(
            [I.parse_input() for I in batch]
        )
        shift_labels, labels = self.prepare_output(
            [I.parse_output() for I in batch]
        )
        cross_attn_mask = self.build_cross_attn_mask(
            input_dict.input_ids,
            labels
        )
        input_emb, memory, num_emb, voc_mask = self.encode(input_dict, num_ids)

        logits, _ = self.decode(
            decoder_input_ids=shift_labels,
            past_value=input_emb,
            num_emb=num_emb,
            voc_mask=voc_mask,
            memory=memory,
            cross_attn_mask=cross_attn_mask
        )

        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, labels, ignore_index=-1)

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            Acc = torch.sum(preds == labels) / (torch.sum(labels != -1) + 1e-5)

        return loss, Acc

    @torch.no_grad()
    def generate_one_step(
        self, 
        input_text: str, 
        max_length: int = 35,
        prev_past_value: Optional[Tensor] = None,
        debug: bool = False
    ) -> str:

        F = {
            "start": dict(
                [(self.voc2dvoc[k], "arg") for k in self.num_token_ids] + \
                [(self.voc2dvoc[k], "arg") for k in self.const_num_token_ids] + \
                [(self.dvoc_token_id["("], "lb")]   # "("
            ),
            "arg": dict(
                [(self.voc2dvoc[k], "op") for k in self.op_token_ids[:-2]] + \
                [(self.dvoc_token_id["[eos0]"], "end")] + \
                [(self.dvoc_token_id["[eos1]"], "end")] + \
                [(self.dvoc_token_id[")"], "rb")]  # ")"
            ),
            "op": dict(
                [(self.voc2dvoc[k], "arg") for k in self.num_token_ids] + \
                [(self.voc2dvoc[k], "arg") for k in self.const_num_token_ids] + \
                [(self.dvoc_token_id["("], "lb")]   # "("
            ),
            "lb": dict(
                [(self.voc2dvoc[k], "arg") for k in self.num_token_ids] + \
                [(self.voc2dvoc[k], "arg") for k in self.const_num_token_ids]
            ),
            "rb": dict(
                [(self.voc2dvoc[k], "op") for k in self.op_token_ids[:-2]] + \
                [(self.dvoc_token_id["[eos0]"], "end")] + \
                [(self.dvoc_token_id["[eos1]"], "end")] + \
                [(self.dvoc_token_id[")"], "rb")]  # ")"
            )
        }
        allowed_token_ids = {
            "start":  list(F["start"].keys()),
            "arg":    list(F["arg"].keys()),
            "op":     list(F["op"].keys()),
            "lb":     list(F["lb"].keys()),
            "rb":     list(F["rb"].keys()),
        }

        input_dict, num_ids = self.prepare_input([input_text])
        input_emb, memory, num_emb, voc_mask = self.encode(input_dict, num_ids)

        past_value = input_emb if prev_past_value is None else prev_past_value
        decoder_input_ids = torch.tensor(
            self.dvoc_token_id["[eos0]"], 
            dtype=torch.long,
            device=past_value.device
        ).view(1, 1)

        predict_ids = []

        state = "start"
        bracket_count = 0

        lb_id = self.dvoc_token_id["("]
        rb_id = self.dvoc_token_id[")"]
        eos0_id = self.dvoc_eos0_token_id
        eos1_id = self.dvoc_eos1_token_id

        while len(predict_ids) < max_length:
            _allowed = deepcopy(allowed_token_ids[state])
            if bracket_count == 0 and rb_id in _allowed:
                _allowed.remove(rb_id)
            if bracket_count > 0:
                if eos0_id in _allowed:
                    _allowed.remove(eos0_id)
                if eos1_id in _allowed:
                    _allowed.remove(eos1_id)
            if not self.cfg.use_bracket:
                if lb_id in _allowed:
                    _allowed.remove(lb_id)
                if rb_id in _allowed:
                    _allowed.remove(rb_id)
            state_mask = torch.zeros_like(voc_mask, dtype=torch.bool)
            state_mask[0, _allowed] = True

            next_token_logits, past_value = self.decode(
                decoder_input_ids=decoder_input_ids[:, -1],
                past_value=past_value,
                num_emb=num_emb,
                voc_mask=voc_mask & state_mask,
                memory=memory
            )

            next_token_id = torch.argmax(next_token_logits, dim=1, keepdim=True)  # [1, 1]
            predict_ids.append(next_token_id.item())

            if predict_ids[-1] in [eos0_id, eos1_id]:
                break
            else:
                state = F[state][next_token_id.item()]
                if next_token_id.item() == lb_id:
                    bracket_count += 1
                if next_token_id.item() == rb_id:
                    bracket_count -= 1
                decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)  # [1, L] -> [1, L + 1]
        
        predict_tokens = self.tok.convert_ids_to_tokens(
            [self.dvoc2voc[x] for x in predict_ids]
        )
        predict_text = "".join(predict_tokens)\
            .replace("[eos0]", "")\
            .replace("[eos1]", "")
        exit_flag = (len(predict_ids) == 0 or predict_ids[-1] == self.dvoc_eos1_token_id)
        return predict_text, exit_flag, past_value

    def parse_opSeq(self, op_text: str, arg0: int) -> Op:
        expr_toks = [t for t in re.split(r"([\*\/\^\+\-\(\)])", op_text) if t != ""]
        expr_str = op_text
        return OpSeq(arg0=arg0, expr_toks=expr_toks, expr_str=expr_str)

    @torch.no_grad()
    def generate(
        self, 
        question: str, 
        nums: List[str], 
        const_nums: List[str],
        debug: bool = False
    ) -> List[OpSeq]:
        opSeq_list: List[OpSeq] = []
        nums_size = len(nums)
        prev_past_value = None
        step = 0
        while len(nums) + len(opSeq_list) < self.cfg.max_nums_size and step < self.cfg.max_step_size:
            I = OpSeqDataInstance(
                question=question,
                nums=nums,
                const_nums=const_nums,
                opSeq_list=opSeq_list
            )
            op_text, exit_flag, past_value = self.generate_one_step(
                I.parse_input(),
                prev_past_value=prev_past_value,
                debug=debug
            )
            try:
                opSeq = self.parse_opSeq(op_text, nums_size)
                opSeq_list.append(opSeq)
            except:
                break
            nums_size += 1
            prev_past_value = past_value
            step += 1
            if exit_flag:
                break
        return opSeq_list
