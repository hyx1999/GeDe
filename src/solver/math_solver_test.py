# math solver with multidecoder

import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.bert import BertModel, BertTokenizer
from transformers.models.roberta import RobertaModel, RobertaTokenizer
from transformers.models.deberta import DebertaModel, DebertaTokenizer
from transformers.models.xlm_roberta import XLMRobertaModel, XLMRobertaTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union
from tqdm import tqdm

from loguru import logger
from math_utils import Op, Expr, Tok, MathDataInstance
from cfg import MathConfig


model_dict = {
    "model": {
        "bert-base-uncased": BertModel,
        "bert-large-uncased": BertModel,
        "bert-base-chinese": BertModel,
        "roberta-base": RobertaModel,
        "hfl/chinese-roberta-wwm-ext": BertModel,
    },
    "tokenizer": {
        "bert-base-uncased": BertTokenizer,
        "bert-large-uncased": BertTokenizer,
        "bert-base-chinese": BertTokenizer,
        "roberta-base": RobertaTokenizer,
        "hfl/chinese-roberta-wwm-ext": BertTokenizer,
    }
}

class ExprTokenizer:
    
    def __init__(self, 
        quant_size: int, 
        const_quant_size: int, 
        ext_tokens: List[str] = None
    ) -> None:
        if ext_tokens is None:
            ext_tokens = []
        self.bos_token = "[bos]"
        self.eos_token = "[eos]"
        self.pad_token = "[pad]"
        self.null_token = "[null]"
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.null_token_id = 3
        self.tokens = \
            [self.bos_token, self.eos_token, self.pad_token] + \
            ["+", "-", "*", "/"] + ext_tokens + \
            [f"[c{i}]" for i in range(const_quant_size)] + \
            [f"[num{i}]" for i in range(quant_size)]
        self.token_id = {
            token: index for index, token in enumerate(self.tokens)
        }
        self.qunat_id = {
            i: self.token_id[f"[num{i}]"] for i in range(quant_size)
        }
        self.voacb_size = len(self.tokens)

    def __call__(self, 
        batch_text: List[str], 
        padding: bool = True
    ) -> List[List[int]]:
        batch_result: List[List[int]] = []
        for text in batch_text:
            result = []
            text_length = len(text)
            index = 0
            while index < text_length:
                if text[index] == " ":
                    index += 1
                    continue
                match_token = None
                for token in self.tokens:   # 按理说需要按照长度排序, 这里没有单词是前缀的情况
                    if text[index:index + len(token)] == token:
                        match_token = token
                        index += len(token)
                        break
                assert match_token is not None
                result.append(self.token_id[match_token])
            batch_result.append(result)
        max_length = max(len(x) for x in batch_result)
        if padding:
            for result in batch_result:
                delta_length = max_length - len(result)
                result.extend([self.token_id[self.pad_token]] * delta_length)
        return batch_result

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.tokens[ids]
        else:
            return [self.tokens[x] for x in ids]
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.token_id[tokens]
        else:
            return [self.token_id[x] for x in tokens]


class AttentionOutput(nn.Module):
    
    def __init__(self,
        hidden_dim: int
    ) -> None:
        super().__init__()
        self.dense0 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, attention_output: Tensor) -> Tensor:
        intermeidate = self.act(self.dense0(attention_output))
        output = self.norm(self.dense1(intermeidate) + attention_output)
        return output


class Attention(nn.Module):

    def __init__(self,
        hidden_dim: int
    ) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim ** 0.5)
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)        
        self.output = AttentionOutput(hidden_dim)

    def forward(self, 
        input_states: Tensor,
        memory_states: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        q = self.query(input_states)   # [N, L1, H]
        k = self.key(memory_states)    # [N, L2, H]
        v = self.value(memory_states)  # [N, L2, H]

        attn_weight = torch.einsum('bik,bjk->bij', q, k)  # [N, L1, L2]
        attn_weight = attn_weight / self.scale  # [QK^T/sqrt(d_k)]
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(1)
            attn_mask = (1.0 - attn_mask.float()) * torch.finfo(torch.float).min
            attn_weight = attn_weight + attn_mask
        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_output = torch.einsum('bij,bjk->bik', attn_weight, v)

        attn_output = self.norm(self.dense(attn_output) + input_states)
        
        return self.output(attn_output)


class MathDecoder(nn.Module):

    def __init__(self,
        cfg: MathConfig,
        quant_size: int,
        vocab_size: int,
        hidden_dim: int, 
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant_size = quant_size
        self.vocab_size = vocab_size
        
        self.fix_states = nn.Embedding(vocab_size - quant_size, hidden_dim)
        self.mix_ffn = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mem_attn   = Attention(hidden_dim)
        self.quant_attn = Attention(hidden_dim)

        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def prepare_word_states(self,
        quant_states: Tensor
    ):
        fix_states = self.fix_states.weight.unsqueeze(0).repeat([quant_states.shape[0], 1, 1])
        word_states = torch.cat((fix_states, quant_states), dim=1)  # [N, |W|, H]
        return word_states
    
    def embedding(self,
        input_ids: Tensor,
        word_states: Tensor,
    ):
        input_ids = input_ids.unsqueeze(-1).repeat([1, 1, word_states.shape[-1]])
        input_states = torch.gather(word_states, dim=1, index=input_ids)
        return input_states
        
    def compute_logits(self, 
        output_states: Tensor, 
        word_states: Tensor,
        vocab_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output_states = self.transform(output_states)  # [N, L, H]
        # word_states  # [N, |W|, H]
        logits = torch.einsum('bik,bjk->bij', output_states, word_states)
        if vocab_mask is not None:
            if len(vocab_mask.shape) == 2:
                vocab_mask = vocab_mask.unsqueeze(1)
            vocab_mask = (1.0 - vocab_mask.float()) * torch.finfo(torch.float).min
            logits = logits + vocab_mask
        return logits
    
    def forward(
        self,
        decoder_input_ids: Tensor,  # [N, L1]
        mixin_id_ids: Tensor,  # [N, L1]
        hidden_state: Tensor,  # [N, H],
        quant_states: Tensor,  # [N, |Q|, H],
        quant_mask: Tensor,  # [N, L, |Q|]
        memory_states: Tensor,
    ) -> Tensor:
        word_states = self.prepare_word_states(quant_states)
        input_states = self.embedding(decoder_input_ids, word_states)
        mixin_states = self.embedding(mixin_id_ids, word_states)
        
        input_states = self.mix_ffn(
            torch.cat((input_states,mixin_states), dim=-1)
        )
        inter_states, output_hidden_state = self.gru(input_states, hidden_state.unsqueeze(dim=0))
        inter_states = self.mem_attn(inter_states, memory_states)
        output_states = self.quant_attn(inter_states, quant_states, quant_mask)

        fix_mask   = torch.ones(quant_mask.shape[:2] + (self.vocab_size - self.quant_size,), dtype=torch.bool, device=self.cfg.device)
        vocab_mask = torch.cat((fix_mask, quant_mask), dim=-1)
        
        logits = self.compute_logits(output_states, word_states, vocab_mask)
        
        return logits, output_hidden_state.squeeze(dim=0)


class MathRepresenter(nn.Module):
    
    def __init__(self,
        cfg: MathConfig,
        quant_size: int,
        vocab_size: int,
        hidden_dim: int, 
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant_size = quant_size
        self.vocab_size = vocab_size
        
        self.fix_states = nn.Embedding(vocab_size - quant_size, hidden_dim)
        self.quant_id_embedding = nn.Embedding(self.cfg.quant_size, hidden_dim)
        self.position_embedding = nn.Embedding(self.cfg.expr_size , hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mem_attn   = Attention(hidden_dim)
        self.quant_attn = Attention(hidden_dim)

    def embedding(self,
        input_ids: Tensor, 
        quant_states: Tensor
    ) -> Tensor:
        input_ids = input_ids.unsqueeze(-1).repeat([1, 1, quant_states.shape[-1]])
        fix_states = self.fix_states.weight.unsqueeze(0).repeat([quant_states.shape[0], 1, 1])
        word_states = torch.cat((fix_states, quant_states), dim=1)  # [N, |W|, H]
        input_states = torch.gather(word_states, dim=1, index=input_ids)
        return input_states
    
    def forward(self, 
        decoder_input_ids: Tensor,
        hidden_state: Tensor,
        position_ids: Tensor,
        quant_id_ids: Tensor,
        quant_states: Tensor,
        quant_mask: Tensor,
        memory_states: Tensor,
    ) -> Tensor:
        input_states = self.embedding(decoder_input_ids, quant_states)
        input_states = input_states + self.position_embedding(position_ids) + self.quant_id_embedding(quant_id_ids)
        inter_states, _ = self.gru(input_states, hidden_state.unsqueeze(0))
        inter_states = self.mem_attn(inter_states, memory_states)
        output_states = self.quant_attn(inter_states, quant_states, quant_mask)
        return output_states


class MathEncoder(nn.Module):

    def __init__(self, cfg: MathConfig, model_name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.bert = model_dict["model"][model_name].from_pretrained(
            model_name,
            cache_dir="../cache/model"
        )
    
    def forward(
        self,
        input_dict: Dict[str, Tensor]
    ) -> Tensor:
        encoder_outputs = self.bert(**input_dict).last_hidden_state  # [N, L, D]
        return encoder_outputs


class MathSolverTest(nn.Module):
    
    def __init__(
        self,
        cfg: MathConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        model_name = self.cfg.model_name
                  
        self.lang_tok = model_dict["tokenizer"][model_name].from_pretrained(
            self.cfg.model_name,
            cache_dir="../cache/model"
        )
        self.expr_tok = ExprTokenizer(
            self.cfg.quant_size,
            self.cfg.const_quant_size,
            self.cfg.ext_tokens
        )
        self.encoder = MathEncoder(
            cfg,
            self.cfg.model_name
        )
        self.representer = MathRepresenter(
            cfg,
            self.cfg.quant_size,
            self.expr_tok.voacb_size,
            self.encoder.bert.config.hidden_size,
        )
        self.decoder = MathDecoder(
            cfg,
            self.cfg.quant_size,
            self.expr_tok.voacb_size,
            self.encoder.bert.config.hidden_size,
        )

        self.update_vocab(cfg.ext_tokens)
        self.encoder.bert.resize_token_embeddings(len(self.lang_tok))
        
    def save_model(self, dir_path: str, suffix: str = "") -> None:
        path = os.path.join(dir_path, f"model_{suffix}.pth")
        torch.save(self.state_dict(), path)
         
    def load_model(self, dir_path: str, suffix: str = "") -> None:
        path = os.path.join(dir_path, f"model_{suffix}.pth")
        self.load_state_dict(torch.load(path))

    def update_vocab(self, ext_tokens: Optional[List[str]]):
        if ext_tokens is None:
            ext_tokens = []
        tokens = \
            ["+", "-", "*", "/"] + ext_tokens + \
            ['[int]', '[float]', '[frac]', '[perc]'] + \
            [f"[c{i}]"   for i in range(self.cfg.const_quant_size)] + \
            [f"[num{i}]" for i in range(self.cfg.quant_size)] + \
            [f'[rk{i}]'  for i in range(self.cfg.quant_size)]
        self.lang_tok.add_tokens(tokens)
        
        self.quant_tokens_id = list(self.lang_tok.convert_tokens_to_ids(
            [f"[num{i}]" for i in range(self.cfg.quant_size)]
        ))

    def prepare_input(
        self,
        input_text: List[str]
    ) -> Tuple[Dict[str, Tensor], List[List[int]]]:
        input_dict = self.lang_tok(
            input_text, 
            return_tensors="pt", 
            padding=True
        ).to(self.cfg.device)
        batch_quant_ids   = []
        for ids in input_dict.input_ids:
            quant_map = {}
            quant_ids = []
            for i, token_id in enumerate(ids):
                token_id = token_id.item()
                if token_id in self.quant_tokens_id:
                    num_id = self.quant_tokens_id.index(token_id)
                    if num_id not in quant_map:
                        quant_map[num_id] = i
            for i in range(len(quant_map)):
                quant_ids.append(quant_map[i])
            batch_quant_ids.append(quant_ids)

        return input_dict, batch_quant_ids
    
    def prepare_output(
        self, 
        output_text: List[str],
        quant_num: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tuple[int, int, int]]]:
        batch_labels: List[List[int]] = self.expr_tok(
            output_text, 
            padding=True,
        )
        pt_labels = torch.tensor(
            batch_labels,
            dtype=torch.long,
            device=self.cfg.device
        )
        shift_pt_labels = torch.zeros_like(pt_labels)
        shift_pt_labels[:, 1:].copy_(pt_labels[:, :-1])
        shift_pt_labels[:, 0] = self.expr_tok.bos_token_id
        pt_labels[pt_labels == self.expr_tok.pad_token_id] = -1
        
        pt_mixin_id = torch.zeros_like(shift_pt_labels)
        pt_mixin_id.fill_(self.expr_tok.null_token_id)
        for i in range(pt_mixin_id.shape[0]):
            delta = 0
            for j in range(1, pt_mixin_id.shape[1]):
                if shift_pt_labels[i, j].item() == self.expr_tok.bos_token_id:
                    pt_mixin_id[i, j] = self.expr_tok.qunat_id[quant_num[i] + delta]
                    delta += 1
        
        batch_sep = []
        for labels in batch_labels:
            sep = []
            for i in range(1, len(labels)):
                if labels[i] in [self.expr_tok.bos_token_id, self.expr_tok.eos_token_id]:
                    sep.append(i + 1)
            batch_sep.append(sep)
        batch_quant_id  = []
        batch_position  = []
        batch_quant_rng = []
        for i, sep in enumerate(batch_sep):
            quant_id = []
            position = []
            quant_rng = []
            last_x = 0
            for j, x in enumerate(sep):
                quant_id.extend([quant_num[i] + j] * (x - last_x))
                position.extend(list(range(x - last_x)))
                quant_rng.append((quant_num[i] + j, last_x, x))
                last_x = x
            quant_id.extend([0] * (len(batch_labels[i]) - len(quant_id)))
            position.extend([0] * (len(batch_labels[i]) - len(position)))
            batch_quant_id.append(quant_id)
            batch_position.append(position)
            batch_quant_rng.append(quant_rng)
        pt_quant_id = torch.tensor(batch_quant_id, dtype=torch.long, device=self.cfg.device)
        pt_position = torch.tensor(batch_position, dtype=torch.long, device=self.cfg.device)
        pt_range = torch.arange(self.cfg.quant_size, dtype=torch.long, device=self.cfg.device)\
            .expand(*(pt_quant_id.shape + (-1,)))  # [B, L, |Q|]
        pt_quant_mask = (pt_range < pt_quant_id.unsqueeze(-1))
        
        return pt_labels, shift_pt_labels, pt_mixin_id, pt_quant_id, pt_position, pt_quant_mask, batch_quant_rng
    
    def encode(
        self, 
        input_dict: Dict[str, Tensor], 
        batch_quant_ids: List[List[int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(batch_quant_ids)
        memory_states: Tensor = self.encoder(input_dict)
        quant_states  = torch.zeros(
            batch_size, self.cfg.quant_size, self.encoder.bert.config.hidden_size, dtype=torch.float, device=memory_states.device)
        for i in range(batch_size):
            n = len(batch_quant_ids[i])
            quant_states[i, :n, :].copy_(memory_states[i, batch_quant_ids[i], :])
        hidden_state = memory_states[:, 0, :].clone()
        return hidden_state, memory_states, quant_states

    def represent(
        self,
        hidden_state: Tensor,
        decoder_input_ids: Tensor,
        position_ids: Tensor,
        quant_id_ids: Tensor,
        quant_states: Tensor,
        quant_mask: Tensor,
        memory_states: Tensor,
        batch_quant_rng: List[List[Tuple[int, int, int]]]
    ):
        memory_states = self.representer(
            decoder_input_ids=decoder_input_ids,
            hidden_state=hidden_state,
            position_ids=position_ids,
            quant_id_ids=quant_id_ids,
            quant_states=quant_states,
            quant_mask=quant_mask,
            memory_states=memory_states,
        )
        batch_size = decoder_input_ids.shape[0]
        new_quant_states = torch.zeros(
            batch_size, self.cfg.quant_size, self.encoder.bert.config.hidden_size, dtype=torch.float, device=memory_states.device)
        for i, quant_rng in enumerate(batch_quant_rng):
            for j in range(len(quant_rng)):
                quant_id, left, right = quant_rng[j]
                indices = list(range(left, right))
                new_quant_states[i, quant_id, :].copy_(torch.mean(memory_states[i, indices, :], dim=0))
        return quant_states + new_quant_states

    def decode(
        self,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.decoder(*args, **kwargs)
    
    def forward(self, batch: List[MathDataInstance]) -> Tensor:
        input_dict, quant_ids = self.prepare_input(
            [I.parse_input(sep_token="", use_expr=False) for I in batch]
        )
        labels, decoder_input_ids, mixin_id_ids, quant_id_ids, position_ids, quant_mask, batch_quant_rng = \
            self.prepare_output(
                [I.parse_output(self.expr_tok.bos_token, self.expr_tok.eos_token) for I in batch],
                [len(x) for x in quant_ids]
            )
        hidden_state, memory_states, quant_states = self.encode(input_dict, quant_ids)

        quant_states = self.represent(
            decoder_input_ids=decoder_input_ids,
            hidden_state=hidden_state,
            quant_id_ids=quant_id_ids,
            position_ids=position_ids,
            quant_states=quant_states,
            quant_mask=quant_mask,
            memory_states=memory_states,
            batch_quant_rng=batch_quant_rng
        )
        
        logits, _ = self.decode(
            decoder_input_ids=decoder_input_ids,
            mixin_id_ids=mixin_id_ids,
            hidden_state=hidden_state,
            quant_states=quant_states,
            quant_mask=quant_mask,
            memory_states=memory_states,
        )
        
        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, labels, ignore_index=-1)

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            Acc = torch.sum(preds == labels) / (torch.sum(labels != -1) + 1e-5)

        return loss, Acc


    @torch.no_grad()
    def expr_beam_search(
        self,
        hidden_state: Tensor, 
        memory_states: Tensor, 
        quant_states: Tensor,
        quant_num_init: int,
        expr_list: List[Expr],
        gru_hidden_state: Tensor,
        max_length: int = 4,
        beam_size: int = 4,
    ):

        if len(expr_list) > 0:
            inter_text = " ".join(sum([expr.expr_toks + [self.expr_tok.bos_token] for expr in expr_list], []))
            _, decoder_input_ids, _, quant_id_ids, position_ids, quant_mask, batch_quant_rng = \
                self.prepare_output([inter_text], [quant_num_init])
            quant_states = self.represent(
                decoder_input_ids=decoder_input_ids,
                hidden_state=hidden_state,
                quant_id_ids=quant_id_ids,
                position_ids=position_ids,
                quant_states=quant_states,
                quant_mask=quant_mask,
                memory_states=memory_states,
                batch_quant_rng=batch_quant_rng
            )

        quant_num = quant_num_init + len(expr_list)
        search_quant_mask = torch.zeros((1, 1, self.cfg.quant_size), dtype=torch.bool, device=self.cfg.device)
        search_quant_mask[..., :quant_num] = True        

        decoder_input_id = torch.tensor(
            self.expr_tok.bos_token_id,
            dtype=torch.long,
            device=hidden_state.device
        ).view(1, 1)

        mixin_id = torch.tensor(
            self.expr_tok.qunat_id[quant_num],
            dtype=torch.long,
            device=hidden_state.device
        ).view(1, 1)

        mixin_null_id = torch.tensor(
            self.expr_tok.null_token_id,
            dtype=torch.long,
            device=hidden_state.device
        ).view(1, 1)
        
        if gru_hidden_state is None:
            gru_hidden_state = hidden_state
        
        beams = [ExprBeam([], decoder_input_id, mixin_id, gru_hidden_state, 0.0)]
        
        while len(beams) > 0:
            do_search = False
            for beam in beams:
                do_search |= (not beam.end)
            if not do_search:
                break
            next_beams: List[ExprBeam] = []
            for beam in beams:
                if beam.end:
                    next_beams.append(beam)
                else:
                    next_token_logits, next_hidden_state = self.decode(
                        decoder_input_ids=beam.decoder_input_id,
                        mixin_id_ids=beam.mixin_id,
                        hidden_state=beam.gru_hidden_state,
                        quant_states=quant_states,
                        quant_mask=search_quant_mask,
                        memory_states=memory_states,
                    )
                    next_token_probs = torch.log_softmax(next_token_logits.view(-1), dim=0)
                    log_probs, indices = torch.topk(next_token_probs, beam_size)
                    for log_prob, index in zip(log_probs, indices):
                        next_beams.append(
                            beam.extend(log_prob, next_hidden_state, index, mixin_null_id)
                        )
            filtered_beams: List[ExprBeam] = []
            for beam in next_beams:
                tokens = self.expr_tok.convert_ids_to_tokens(beam.predict_ids)
                if not grammar_test(tokens, max_length, self.expr_tok.bos_token, self.expr_tok.eos_token):
                    continue
                if not constraint_test(tokens):  # common constraint
                    continue
                if not beam.end and tokens[-1] in [self.expr_tok.bos_token, self.expr_tok.eos_token]:
                    beam.end = True
                filtered_beams.append(beam)
            beams = filtered_beams
            beams = sorted(beams, key=lambda b: b.score, reverse=True)[:beam_size]

        return beams


    @torch.no_grad()
    def beam_search(
        self, 
        question: str, 
        nums: List[str], 
        const_nums: List[str],
        beam_size: int = 4
    ) -> List[Expr]:
        I0 = MathDataInstance(
            question=question,
            nums=nums,
            const_nums=const_nums,
            expr_list=[]
        )
        input_dict, quant_ids = self.prepare_input([I0.parse_input("", use_expr=False)])        
        hidden_state, memory_states, quant_states = self.encode(input_dict, quant_ids)
        quant_num_init = len(quant_ids[0])
        
        beams: List[StatBeam] = [StatBeam(None, [], 0.0, self.expr_tok.bos_token)]
        while len(beams) > 0:
            do_search = False
            for beam in beams:
                do_search |= (not beam.end)
            if not do_search:
                break
            next_beams: List[StatBeam] = []
            for beam in beams:
                if beam.end:
                    next_beams.append(beam)
                else:
                    expr_beams: List[ExprBeam] = self.expr_beam_search(
                        hidden_state, 
                        memory_states, 
                        quant_states,
                        quant_num_init,
                        beam.expr_list,
                        beam.gru_hidden_state,
                        beam_size=beam_size,
                    )
                    for expr_beam in expr_beams:
                        tokens = self.expr_tok.convert_ids_to_tokens(expr_beam.predict_ids)
                        arg0 = len(nums) + len(beam.expr_list)
                        expr = Expr(arg0=arg0, expr_toks=tokens[:-1], expr_str="".join(tokens[:-1]))
                        end_token = tokens[-1]
                        next_beams.append(beam.extend(gru_hidden_state=expr_beam.gru_hidden_state, expr=expr, score=expr_beam.score, end_token=end_token))
            filtered_beams: List[StatBeam] = []
            for beam in next_beams:
                if (len(nums) + len(beam.expr_list) >= self.cfg.max_nums_size or len(beam.expr_list) >= self.cfg.max_step_size) and beam.end_token != self.expr_tok.eos_token:
                    continue
                if not beam.end and beam.end_token == self.expr_tok.eos_token:
                    beam.end = True
                filtered_beams.append(beam)
            beams = filtered_beams

            # if self.cfg.debug:
            #     print(">>>")
            #     for beam in beams:
            #         print("beam score:", beam.score)
            #         print("beam end_token:", beam.end_token)
            #         print("beam end:", beam.end)
            #         print("beam expr-list:", [" ".join(x.expr_toks) for x in beam.expr_list])

            beams = sorted(beams, key=lambda b: b.score, reverse=True)[:beam_size]

        # if self.cfg.debug:
        #     for beam in beams:
        #         print("final beam score:", beam.score)
        #         print("final beam expr-list:", [" ".join(x.expr_toks) for x in beam.expr_list])
        if len(beams) == 0:
            return None
        return beams[0].expr_list


class ExprBeam:
    
    def __init__(self,
        predict_ids: List[int],
        decoder_input_id: Tensor,
        mixin_id: Tensor,
        gru_hidden_state: Tensor,
        score: float
    ) -> None:
        self.predict_ids = predict_ids
        self.decoder_input_id = decoder_input_id
        self.mixin_id = mixin_id
        self.gru_hidden_state = gru_hidden_state
        self.score = score
        self.end = False
    
    def extend(self, log_prob: Tensor, gru_hidden_state: Tensor, index: Tensor, mixin_id: Tensor) -> 'ExprBeam':
        length = len(self.predict_ids)
        next_beam = ExprBeam(
            predict_ids=self.predict_ids + [index.item()],
            decoder_input_id=index.clone().view(1, 1),
            mixin_id=mixin_id.clone().view(1, 1),
            gru_hidden_state=gru_hidden_state.clone(),
            score=(self.score * length + log_prob) / (length + 1)
        )
        return next_beam


class StatBeam:
    
    def __init__(self,
        gru_hidden_state: Tensor,
        expr_list: List[Expr],
        score: float,
        end_token: str
    ) -> None:
        self.gru_hidden_state = gru_hidden_state
        self.expr_list = expr_list
        self.score = score
        self.end_token = end_token
        self.end = False
    
    def extend(self, gru_hidden_state: Tensor, expr: Expr, score: float, end_token: str) -> 'StatBeam':
        length = len(self.expr_list)
        next_beam = StatBeam(
            gru_hidden_state=gru_hidden_state.clone(),
            expr_list=self.expr_list + [expr],
            score=(self.score * length + score) / (length + 1),
            end_token=end_token
        )
        return next_beam


def grammar_test(tokens: List[Tok], max_length: int, bos_token: str, eos_token: str) -> bool:
    if len(tokens) >= max_length and tokens[-1] not in [bos_token, eos_token]:
        return False
    end = tokens[-1] in [bos_token, eos_token]
    if end:
        tokens = tokens[:-1]
    pat = re.compile("\[num\d+\]|\[c\d+\]")
    n_stk = []
    o_stk = []
    try:
        for i, tok in enumerate(tokens):
            if pat.match(tok):
                if i > 0 and tokens[i - 1] not in '+-*/^(':
                    return False
                n_stk.append('n')
            elif tok in '+-*/^':
                if i > 0 and tokens[i - 1] in '+-*/^(':
                    return False
                o_stk.append(tok)
        if end:
            while len(o_stk) > 0:
                o_stk.pop()
                n_stk.pop()
                n_stk.pop()
                n_stk.append('n')
            if len(n_stk) != 1:
                return False
    except:
        return False 
    return True


def constraint_test(tokens: List[Tok]):
    if len(tokens) < 3:
        return True
    if tokens[0] == tokens[2] and tokens[1] in "-/^":
        return False
    return True
