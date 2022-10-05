import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.bert import BertModel, BertTokenizer, BertConfig
from transformers.models.bart import BartModel, BartTokenizer, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

import os
from copy import deepcopy
from typing import Dict, List, Any, AnyStr, Set, Optional, Tuple, Union
from tqdm import tqdm

from loguru import logger
from kbqa_utils import Expr, RawDataInstance, DataBatch, KBQADataset, KBClient
from cfg import KBQAConfig


class KBQASolver(nn.Module):
    
    def __init__(self, 
        cfg_dict: Dict[str, Any],
        domain_info: Set[str],
        extra_tokens: List[str]
    ) -> None:
        super().__init__()
        self.cfg = KBQAConfig(**cfg_dict)
        self.kb = KBClient()
        self.domain_info: Set[str] = domain_info
        self.extra_tokens = extra_tokens

        self.tokenizer = BartTokenizer.from_pretrained(
            self.cfg.model_name,
            cache_dir="../cache/model"
        )
        self.bart = BartModel.from_pretrained(
            self.cfg.model_name,
            cache_dir="../cache/model"
        )
    
    def encode_rels(self):
        ...
    
    def encode(self):
        ...
    
    def decode(self):
        ...
        
    def forward(self, batch: DataBatch):
        ...
    
    @torch.no_grad()
    def sub_generate(self):
        ...
    
    @torch.no_grad()
    def generate(self):
        ...
