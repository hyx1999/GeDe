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
from transformers import BartForConditionalGeneration, BartTokenizer

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union
from tqdm import tqdm

from loguru import logger
from math_utils import Expr, Tok, MathDataInstance, TemplateDataInstance
from cfg import MathConfig


model_dict = {
    "model": {
        "facebook/bart-base": BartForConditionalGeneration
    },
    "tokenizer": {
        "facebook/bart-base": BartTokenizer
    }
}

class MathSolverBART(nn.Module):
    
    def __init__(
        self,
        cfg: MathConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        model_name = self.cfg.model_name
        
        self.tok: BartTokenizer = model_dict["tokenizer"][model_name].from_pretrained(
            self.cfg.model_name,
            cache_dir="../cache/model",
        )
        self.model: BartForConditionalGeneration = model_dict["model"][model_name].from_pretrained(
            self.cfg.model_name,
            cache_dir="../cache/model",
        )

        self.update_vocab(cfg.ext_tokens)
        self.model.resize_token_embeddings(len(self.tok))
        
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
                "[pow]",
                "[->]"
            ] + ext_tokens + \
            [f"[c{i}]"   for i in range(self.cfg.const_quant_size)] + \
            [f"[num{i}]" for i in range(self.cfg.quant_size)]
        self.tok.add_tokens(tokens)

        self.quant_tokens_id = list(self.tok.convert_tokens_to_ids(
            [f"[num{i}]" for i in range(self.cfg.quant_size)]
        ))

    def prepare_input(
        self,
        input_text: List[str]
    ) -> Tuple[Dict[str, Tensor], List[List[int]]]:
        input_dict = self.tok(
            input_text, 
            return_tensors="pt", 
            padding=True
        ).to(self.cfg.device)
        return input_dict
    
    def prepare_output(
        self, 
        output_text: List[str]
    ) -> Tuple[Tensor]:
        labels: List[List[int]] = self.tok(
            output_text, 
            return_tensors="pt",
            padding=True,
        ).input_ids.to(self.cfg.device)
        return labels
    
    def forward(self, batch: List[TemplateDataInstance]) -> Tensor:
        inputs = self.prepare_input(
            [I.parse_input("#", use_expr=False) for I in batch]
        )
        labels = self.prepare_output(
            [I.parse_output_bart(self.tok.bos_token, self.tok.eos_token, erase_end=True) for I in batch]
        )
        
        outputs = self.model(**inputs, labels=labels)
        
        logits = outputs.logits
        loss = outputs.loss

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=-1)
            acc_score = torch.sum(preds == labels) / (torch.sum(labels != -1) + 1e-5)

        return loss, acc_score
