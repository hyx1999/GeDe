# math solver with multidecoder

from lib2to3.pgen2 import token
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
from math_utils import MultiExpr, Tok, TemplateDataInstance
from cfg import MathConfig


model_dict = {
    "model": {
        "bert-base-uncased": BertModel,
        "bert-large-uncased": BertModel,
        "bert-base-chinese": BertModel,
        "roberta-base": RobertaModel,
        "roberta-large": RobertaModel,
        "hfl/chinese-roberta-wwm-ext": BertModel,
    },
    "tokenizer": {
        "bert-base-uncased": BertTokenizer,
        "bert-large-uncased": BertTokenizer,
        "bert-base-chinese": BertTokenizer,
        "roberta-base": RobertaTokenizer,
        "roberta-large": RobertaTokenizer,
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
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.tokens = \
            [self.bos_token, self.eos_token, self.pad_token] + \
            [
                "[solve_linear_equation]", 
                "[quadratic_function_integral]", 
                "[quadratic_function_extremum]",
                "[add]",
                "[sub]",
                "[mul]",
                "[div]",
                "[pow]"
            ] + ext_tokens + \
            [f"[c{i}]" for i in range(const_quant_size)] + \
            [f"[num{i}]" for i in range(quant_size)]
        self.sorted_tokens = sorted(self.tokens, key=lambda x: len(x), reverse=True)
        self.token_id = {
            token: index for index, token in enumerate(self.tokens)
        }
        self.voacb_size = len(self.tokens)
            
    def __call__(self, 
        batch_text: List[str], 
        padding: bool = True
    ) -> List[List[int]]:
        # print(batch_text)
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
                for token in self.sorted_tokens:
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
            attn_mask = (1.0 - attn_mask.unsqueeze(1).float()) * torch.finfo(torch.float).min
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
        
        self.attn = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fix_states = nn.Embedding(vocab_size - quant_size, hidden_dim)
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def embedding(self, 
        input_ids: Tensor, 
        quant_states: Tensor
    ) -> Tensor:
        input_ids = input_ids.unsqueeze(-1).repeat([1, 1, quant_states.shape[-1]])
        fix_states = self.fix_states.weight.unsqueeze(0).repeat([quant_states.shape[0], 1, 1])
        word_states = torch.cat((fix_states, quant_states), dim=1)  # [N, |W|, H]
        input_states = torch.gather(word_states, dim=1, index=input_ids)
        return word_states, input_states
    
    def compute_logits(self, 
        output_states: Tensor, 
        word_states: Tensor,
        vocab_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output_states = self.transform(output_states)  # [N, L, H]
        # word_states  # [N, |W|, H]
        logits = torch.einsum('bik,bjk->bij', output_states, word_states)
        if vocab_mask is not None:
            vocab_mask = (1.0 - vocab_mask.unsqueeze(1).float()) * torch.finfo(torch.float).min
            logits = logits + vocab_mask
        return logits
    
    def forward(
        self,
        decoder_input_ids: Tensor,  # [N] or [N, L1]
        hidden_state: Tensor,  # [N, H],
        quant_states: Tensor,  # [N, |Q|, H],
        memory_states: Tensor,  # [N, L', D]
        vocab_mask: Tensor,  # [N, |W|]
        attn_mask: Optional[Tensor] = None,  # [N, L']
    ) -> Tensor:
        if len(decoder_input_ids.shape) == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(1)

        word_states, input_states = self.embedding(decoder_input_ids, quant_states)
        inter_states, output_hidden_state = self.gru(input_states, hidden_state.unsqueeze(dim=0))

        output_states = self.attn(inter_states, memory_states, attn_mask)
        
        logits = self.compute_logits(output_states, word_states, vocab_mask)
        
        return logits, output_hidden_state.squeeze(dim=0)


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


class MathSolverRETemplate(nn.Module):
    
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
            [
                "[solve_linear_equation]", 
                "[quadratic_function_integral]", 
                "[quadratic_function_extremum]",
                "[add]",
                "[sub]",
                "[mul]",
                "[div]",
                "[pow]"
            ] + ext_tokens + \
            [f"[c{i}]"   for i in range(self.cfg.const_quant_size)] + \
            [f"[num{i}]" for i in range(self.cfg.quant_size)]
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
        for j, ids in enumerate(input_dict.input_ids):
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
        output_text: List[str]
    ) -> Tuple[Tensor, Tensor]:
        labels: List[List[int]] = self.expr_tok(
            output_text, 
            padding=True,
        )
        labels = torch.tensor(
            labels,
            dtype=torch.long,
            device=self.cfg.device
        )
        shift_labels = torch.zeros_like(labels)
        shift_labels[:, 1:].copy_(labels[:, :-1])
        shift_labels[:, 0] = self.expr_tok.bos_token_id
        labels[labels == self.expr_tok.pad_token_id] = -1
        return shift_labels, labels
    
    def encode(
        self, 
        input_dict: Dict[str, Tensor], 
        batch_quant_ids: List[List[int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        quant_size = self.cfg.quant_size
        vocab_size = self.expr_tok.voacb_size
        batch_size = len(batch_quant_ids)
        memory_states: Tensor = self.encoder(input_dict)
        quant_states  = torch.zeros(
            batch_size, self.cfg.quant_size, self.encoder.bert.config.hidden_size, dtype=torch.float, device=memory_states.device)
        vocab_mask = torch.zeros(
            batch_size, self.expr_tok.voacb_size, dtype=torch.bool, device=memory_states.device)
        for i in range(batch_size):
            n = len(batch_quant_ids[i])
            vocab_mask[i, :vocab_size - quant_size + n] = True
            quant_states[i, :n, :].copy_(memory_states[i, batch_quant_ids[i], :])
        hidden_state = memory_states[:, 0, :].clone()
        return hidden_state, memory_states, quant_states, vocab_mask

    def decode(
        self,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.decoder(*args, **kwargs)
    
    def forward(self, batch: List[TemplateDataInstance]) -> Tensor:
        input_dict, quant_ids = self.prepare_input(
            [I.parse_input("#") for I in batch]
        )
        shift_labels, labels = self.prepare_output(
            [I.parse_output(self.expr_tok.bos_token, self.expr_tok.eos_token) for I in batch]
        )
        hidden_state, memory_states, quant_states, vocab_mask = self.encode(input_dict, quant_ids)

        logits, _ = self.decode(
            decoder_input_ids=shift_labels,
            hidden_state=hidden_state,
            quant_states=quant_states,
            memory_states=memory_states,
            vocab_mask=vocab_mask,
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
        input_text: str,
        max_length: int = 20,
        beam_size: int = 4,
    ):
        input_dict, quant_ids = self.prepare_input([input_text])
        hidden_state, memory_states, quant_states, vocab_mask = self.encode(input_dict, quant_ids)

        decoder_input_id = torch.tensor(
            self.expr_tok.bos_token_id,
            dtype=torch.long,
            device=hidden_state.device
        ).view(1, 1)
        
        beams = [ExprBeam([], decoder_input_id, hidden_state, 0.0)]
        end_beams = []
        step = 0
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
                        hidden_state=beam.gru_hidden_state,
                        quant_states=quant_states,
                        memory_states=memory_states,
                        vocab_mask=vocab_mask,
                    )
                    next_token_probs = torch.log_softmax(next_token_logits.view(-1), dim=0)
                    log_probs, indices = torch.topk(next_token_probs, beam_size)
                    for log_prob, index in zip(log_probs, indices):
                        next_beams.append(
                            beam.extend(log_prob, next_hidden_state, index)
                        )
            filtered_beams: List[ExprBeam] = []
            for beam in next_beams:
                tokens = self.expr_tok.convert_ids_to_tokens(beam.predict_ids)
                if not grammar_test(tokens, max_length, self.expr_tok.bos_token, self.expr_tok.eos_token):
                    continue
                if not beam.end and tokens[-1] in [self.expr_tok.bos_token, self.expr_tok.eos_token]:
                    beam.end = True
                    end_beams.append(beam)

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
    ) -> List[MultiExpr]:
        beams: List[StatBeam] = [StatBeam([], 0.0, self.expr_tok.bos_token)]
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
                    I = TemplateDataInstance(
                        question=question,
                        nums=nums,
                        const_nums=const_nums,
                        expr_list=beam.expr_list
                    )
                    expr_beams = self.expr_beam_search(
                        I.parse_input("#"),
                        beam_size=beam_size
                    )
                    for expr_beam in expr_beams:
                        tokens = self.expr_tok.convert_ids_to_tokens(expr_beam.predict_ids)
                        start_num = beam.expr_list[-1].args[-1] + 1 if len(beam.expr_list) > 0 else len(nums)
                        args = [start_num + i for i in range(parse_fn_dim(tokens[:-1]))]  # erae bos and eos
                        expr = MultiExpr(args=args, expr_toks=tokens[:-1], expr_str=" ".join(tokens[:-1]))
                        end_token = tokens[-1]
                        next_beams.append(beam.extend(expr=expr, score=expr_beam.score, end_token=end_token))
            filtered_beams: List[StatBeam] = []
            for beam in next_beams:
                if len(beam.expr_list) > 0 and beam.expr_list[-1].args[-1] > self.cfg.quant_size:
                    continue
                if len(beam.expr_list) >= self.cfg.max_step_size and beam.end_token != self.expr_tok.eos_token:
                    continue
                if not beam.end and beam.end_token == self.expr_tok.eos_token:
                    beam.end = True
                filtered_beams.append(beam)
            beams = filtered_beams
            beams = sorted(beams, key=lambda b: b.score, reverse=True)[:beam_size]

        if len(beams) == 0:
            return None
        return beams[0].expr_list


class ExprBeam:
    
    def __init__(self,
        predict_ids: List[int],
        decoder_input_id: Tensor,
        gru_hidden_state: Tensor,
        score: float
    ) -> None:
        self.predict_ids = predict_ids
        self.decoder_input_id = decoder_input_id
        self.gru_hidden_state = gru_hidden_state
        self.score = score
        self.end = False
    
    def extend(self, log_prob: Tensor, gru_hidden_state: Tensor, index: Tensor) -> 'ExprBeam':
        length = len(self.predict_ids)
        next_beam = ExprBeam(
            predict_ids=self.predict_ids + [index.item()],
            decoder_input_id=index.clone().view(1),
            gru_hidden_state=gru_hidden_state.clone(),
            score=(self.score * length + log_prob) / (length + 1)
        )
        return next_beam


class StatBeam:
    
    def __init__(self, 
        expr_list: List[MultiExpr],
        score: float,
        end_token: str
    ) -> None:
        self.expr_list = expr_list
        self.score = score
        self.end_token = end_token
        self.end = False
    
    def extend(self, expr: MultiExpr, score: float, end_token: str) -> 'StatBeam':
        length = len(self.expr_list)
        next_beam = StatBeam(
            expr_list=self.expr_list + [expr],
            score=(self.score * length + score) / (length + 1),
            end_token=end_token
        )
        return next_beam


def parse_quants_test(tokens: List[Tok]):
    pat = re.compile("\[num\d+\]")
    if any(not pat.match(tokens[i]) for i in range(len(tokens))):
        return False
    return True

def parse_fn_test(tokens: List[Tok], end: bool = False):
    if tokens[0] == "[solve_linear_equation]":
        if end:
            return len(tokens) == 7 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[quadratic_function_integral]":
        if end:
            return len(tokens) == 6 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])            
    elif tokens[0] == "[quadratic_function_extremum]":
        if end:
            return len(tokens) == 4 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[add]":
        if end:
            return len(tokens) == 3 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[sub]":
        if end:
            return len(tokens) == 3 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[mul]":
        if end:
            return len(tokens) == 3 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[div]":
        if end:
            return len(tokens) == 3 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])
    elif tokens[0] == "[pow]":
        if end:
            return len(tokens) == 3 and parse_quants_test(tokens[1:])
        else:
            return parse_quants_test(tokens[1:])

def grammar_test(tokens: List[Tok], max_length: int, bos_token: str, eos_token: str) -> bool:
    if len(tokens) >= max_length and tokens[-1] not in [bos_token, eos_token]:
        return False
    if len(tokens) > 0 and tokens[0] not in [
        "[solve_linear_equation]", 
        "[quadratic_function_integral]", 
        "[quadratic_function_extremum]",
        "[add]",
        "[sub]",
        "[mul]",
        "[div]",
        "[pow]",
    ]:
        return False
    end = False
    if tokens[-1] in [bos_token, eos_token]:
        end = True
        tokens = tokens[:-1]  # remove [bos], [eos]
    return parse_fn_test(tokens, end)

def parse_fn_dim(tokens: List[Tok]):
    if tokens[0] == "[solve_linear_equation]":
        return 2
    else:
        return 1
