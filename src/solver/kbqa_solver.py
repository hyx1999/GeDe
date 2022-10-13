import re
import random
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.bert import BertModel, BertTokenizer
from transformers.models.roberta import RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Set, Optional, Tuple, Union
from tqdm import tqdm

from loguru import logger
from kbqa_utils import Expr, RawDataInstance, DataBatch, KBQADataset, DBClient
from cfg import KBQAConfig


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


class LogicTokenizer:
    
    def __init__(self,
        cfg: KBQAConfig,
    ) -> None:
        self.bos_token = "[bos]"
        self.eos_token = "[eos]"
        self.pad_token = "[pad]"
        self.null_token = "[null]"
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.null_token_id = 3

        self.tokens =  [self.bos_token, self.eos_token, self.pad_token, self.null_token] + cfg.ext_tokens + cfg.rels + cfg.types
        self.tokens_sorted = sorted(self.tokens, key=lambda x: len(x), reverse=True)
        self.token_id = {
            token: index for index, token in enumerate(self.tokens)
        }
        self.variable_id = {
            i: self.token_id[f"[v{i}]"] for i in range(cfg.variable_size)
        }
        self.fix_vocab_size = 4 + len(cfg.ext_tokens)
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
                for token in self.tokens_sorted:
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


class KBQADecoder(nn.Module):
    
    def __init__(self,
        cfg: KBQAConfig,
        hidden_dim: int,
        fix_vocab_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.fix_vocab_size = fix_vocab_size
        
        self.fix_states = nn.Embedding(fix_vocab_size, hidden_dim)
        self.mix_ffn = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mem_attn = Attention(hidden_dim)
        self.var_attn = Attention(hidden_dim)

        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def prepare_word_states(self,
        relation_states: Tensor,
        type_states: Tensor,
        variable_states: Tensor,
    ):
        fix_states = self.fix_states.weight.unsqueeze(0).repeat([relation_states.shape[0], 1, 1])
        word_states = torch.cat((fix_states, relation_states, type_states, variable_states), dim=1)  # [N, |W|, H]
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
        relation_states: Tensor,  # [N, |R|, H],
        type_states: Tensor,      # [N, |T|, H]
        variable_states: Tensor,  # [N, |S|, H]
        variable_mask: Tensor,    # [N, [S|]]
        memory_states: Tensor,
    ) -> Tensor:
        word_states = self.prepare_word_states(relation_states, type_states, variable_states)
        input_states = self.embedding(decoder_input_ids, word_states)
        mixin_states = self.embedding(mixin_id_ids, word_states)
        
        input_states = self.mix_ffn(
            torch.cat((input_states,mixin_states), dim=-1)
        )
        inter_states, output_hidden_state = self.gru(input_states, hidden_state.unsqueeze(dim=0))
        inter_states = self.mem_attn(inter_states, memory_states)
        output_states = self.var_attn(inter_states, variable_states, variable_mask)

        fix_mask = torch.ones(
            variable_mask.shape[:2] + (self.fix_vocab_size + relation_states.shape[1] + type_states.shape[1],), 
            dtype=torch.bool, device=self.cfg.device
        )
        vocab_mask = torch.cat((fix_mask, variable_mask), dim=-1)
        
        logits = self.compute_logits(output_states, word_states, vocab_mask)
        
        return logits, output_hidden_state.squeeze(dim=0)


class KBQARepresenter(nn.Module):
    
    def __init__(self,
        cfg: KBQAConfig,
        hidden_dim: int,
        fix_vocab_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.fix_states = nn.Embedding(fix_vocab_size, hidden_dim)
        self.variable_id_embedding = nn.Embedding(cfg.variable_size, hidden_dim)
        self.position_embedding = nn.Embedding(cfg.expr_size, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mem_attn = Attention(hidden_dim)
        self.var_attn = Attention(hidden_dim)

    def embedding(self,
        input_ids: Tensor, 
        relation_states: Tensor,  # [N, |Q|, H],
        type_states: Tensor,
        variable_states: Tensor
    ) -> Tensor:
        input_ids = input_ids.unsqueeze(-1).repeat([1, 1, relation_states.shape[-1]])
        fix_states = self.fix_states.weight.unsqueeze(0).repeat([relation_states.shape[0], 1, 1])
        word_states = torch.cat((fix_states, relation_states, type_states, variable_states), dim=1)  # [N, |W|, H]
        input_states = torch.gather(word_states, dim=1, index=input_ids)
        return input_states
    
    def forward(self, 
        decoder_input_ids: Tensor,
        hidden_state: Tensor,
        position_ids: Tensor,
        variable_id_ids: Tensor,
        relation_states: Tensor,  # [N, |Q|, H],
        type_states: Tensor,
        variable_states: Tensor,
        variable_mask: Tensor,
        memory_states: Tensor,
    ) -> Tensor:
        input_states = self.embedding(decoder_input_ids, relation_states, type_states, variable_states)
        input_states = input_states + self.position_embedding(position_ids) + self.variable_id_embedding(variable_id_ids)
        inter_states, _ = self.gru(input_states, hidden_state.unsqueeze(0))
        inter_states = self.mem_attn(inter_states, memory_states)
        output_states = self.var_attn(inter_states, variable_states, variable_mask)
        return output_states


class KBQAEncoder(nn.Module):
    
    def __init__(self,
        cfg: KBQAConfig
    ) -> None:
        super().__init__()
        self.bert = model_dict["model"][cfg.model_name].from_pretrained(
            cfg.model_name,
            cache_dir="../cache/model"
        )
    
    def forward(
        self,
        input_dict: Dict[str, Tensor]
    ) -> Tensor:
        encoder_outputs = self.bert(**input_dict).last_hidden_state  # [N, L, D]
        return encoder_outputs


class RelationRanker(nn.Module):
    
    def __init__(self,
        cfg: KBQAConfig
    ) -> None:
        super().__init__()
        self.cfg: KBQAConfig = cfg
        self.bert = model_dict[cfg.model_name]["model"].from_pretrained(
            cfg.model_name,
            cache_dir="../cache/model"
        )
    
    def forward(self,
        text_input: Dict[str, Tensor],
        relation_input: Dict[str, Tensor],
        label: Tensor
    ):
        text_states: Tensor = self.bert(**text_input).last_hidden_state          # [B, ..., H]
        relation_states: Tensor = self.bert(**relation_input).last_hidden_state  # [B * L, ..., H]
        
        text_states = text_states.select(1, 0)
        relation_states = relation_states.select(1, 0).view(*(label.shape + (-1,)))  # [B, L, H]
        
        logits = torch.einsum('bh,bjh->bj', text_states, relation_states)
        
        weight = (label != -1).float()
        target = torch.clamp(label.float(), 0.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
        
        return loss
    
    @torch.no_grad()
    def topk(self,
        text_input: Dict[str, Tensor],
        relation_input: Dict[str, Tensor],
        k: int
    ):
        text_states: Tensor = self.bert(**text_input).last_hidden_state          # [1, ..., H]
        relation_states: Tensor = self.bert(**relation_input).last_hidden_state  # [L, ..., H]
        
        text_states = text_states.select(1, 0).squeeze(0)  # [H]
        relation_states = relation_states.select(1, 0)     # [L, H]
        
        logits = torch.einsum('h,jh->j', text_states, relation_states)
        _, indices = torch.topk(logits, k)
        
        return indices


class EncodeEntityFnMixin:
    
    def encode_entity(self, question: str, entities: List[str]):
        cfg: KBQAConfig = getattr(self, "cfg")
        tokenizer: Union[BertTokenizer, RobertaTokenizer] = getattr(self, "lang_tokenizer")
        encoder: Union[BertModel, RobertaModel] = getattr(self, "encoder")
        text = " [entity] ".join([question] + entities)
        ...


class EncodeSchemaFnMixin:
    
    def encode_schema(self, question: str, relations: List[str], types: List[str]):
        cfg: KBQAConfig = getattr(self, "cfg")
        tokenizer: Union[BertTokenizer, RobertaTokenizer] = getattr(self, "lang_tokenizer")
        encoder: Union[BertModel, RobertaModel] = getattr(self, "encoder")
        relation_states = []
        type_states     = []
        for i in range(0, len(question), cfg.relation_bucket_size):
            ...
        ...


class KBQASolver(nn.Module):
    
    def __init__(self, 
        cfg: KBQAConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.db = DBClient()
        self.ranker = RelationRanker(cfg)
        
        self.lang_tokenizer = model_dict["tokenizer"][cfg.model_name].from_pretrained(
            cfg.model_name,
            cache_dir="../cache/model"            
        )
        self.encoder = KBQAEncoder(cfg)

        self.logic_tokenizer = LogicTokenizer(cfg)
        hidden_dim     = self.encoder.bert.config.hidden_size
        fix_vocab_size = self.logic_tokenizer.fix_vocab_size
        self.decoder     = KBQADecoder(cfg, hidden_dim, fix_vocab_size)
        self.representer = KBQARepresenter(cfg, hidden_dim, fix_vocab_size)

        self.update_vocab(cfg.ext_tokens)
        self.encoder.bert.resize_token_embeddings(len(self.lang_tokenizer))

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
            [f'[v{i}]' for i in range(self.cfg.variable_size)] + \
            ['[entity]', '[relation]', '[type]']
        self.lang_tokenizer.add_tokens(tokens)
    
    def prepare_input(
        self,
        input_text: List[str],
    ) -> Tuple[Dict[str, Tensor], List[List[int]]]:
        input_dict = self.lang_tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True
        ).to(self.cfg.device)
        
        return input_dict
    
    def prepare_output(
        self, 
        output_text: List[str],
        variable_num: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tuple[int, int, int]]]:
        batch_labels: List[List[int]] = self.logic_tokenizer(
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
        shift_pt_labels[:, 0] = self.logic_tokenizer.bos_token_id
        pt_labels[pt_labels == self.logic_tokenizer.pad_token_id] = -1
        
        pt_mixin_id = torch.zeros_like(shift_pt_labels)
        pt_mixin_id.fill_(self.logic_tokenizer.null_token_id)
        for i in range(pt_mixin_id.shape[0]):
            delta = 0
            for j in range(1, pt_mixin_id.shape[1]):
                if shift_pt_labels[i, j].item() == self.logic_tokenizer.bos_token_id:
                    pt_mixin_id[i, j] = self.logic_tokenizer.variable_id[variable_num[i] + delta]
                    delta += 1
        
        batch_sep = []
        for labels in batch_labels:
            sep = []
            for i in range(1, len(labels)):
                if labels[i] in [self.logic_tokenizer.bos_token_id, self.logic_tokenizer.eos_token_id]:
                    sep.append(i + 1)
            batch_sep.append(sep)
        batch_variable_id  = []
        batch_position  = []
        batch_variable_rng = []
        for i, sep in enumerate(batch_sep):
            quant_id = []
            position = []
            quant_rng = []
            last_x = 0
            for j, x in enumerate(sep):
                quant_id.extend([variable_num[i] + j] * (x - last_x))
                position.extend(list(range(x - last_x)))
                quant_rng.append((variable_num[i] + j, last_x, x))
                last_x = x
            quant_id.extend([0] * (len(batch_labels[i]) - len(quant_id)))
            position.extend([0] * (len(batch_labels[i]) - len(position)))
            batch_variable_id.append(quant_id)
            batch_position.append(position)
            batch_variable_rng.append(quant_rng)
        pt_variable_id = torch.tensor(batch_variable_id, dtype=torch.long, device=self.cfg.device)
        pt_position = torch.tensor(batch_position, dtype=torch.long, device=self.cfg.device)
        pt_range = torch.arange(self.cfg.quant_size, dtype=torch.long, device=self.cfg.device)\
            .expand(*(pt_variable_id.shape + (-1,)))  # [B, L, |Q|]
        pt_variable_mask = (pt_range < pt_variable_id.unsqueeze(-1))
        
        return pt_labels, shift_pt_labels, pt_mixin_id, pt_variable_id, pt_position, pt_variable_mask, batch_variable_rng
