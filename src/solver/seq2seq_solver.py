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

from utils import Tok
from cfg import Config


class DecoderTokenizer:

    def __init__(self, ext_words: List[Tok], nums_size: int) -> None:
        ext_words = ["[bos]", "[eos]", "[pad]", "[unk]"] + ext_words
        self.ops_size = len(ext_words)
        self.num_size = nums_size
        self.ext_words = ext_words
        self.words_index = {w: i for i, w in enumerate(ext_words)}
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3
    
    def batch_tokenize(self, exprs: List[List[Tok]]) -> Tuple[Tensor, Tensor, Tensor]:
        exprs = [tokens + ["[eos]"] for tokens in exprs]
        max_len = max(len(tokens) for tokens in exprs)
        for tokens in exprs:
            l = len(tokens)
            tokens.extend(["[pad]"] * (max_len - l))
        batch_ids: List[List[int]] = []
        for tokens in exprs:
            ids = []
            for w in tokens:
                m: re.Match = re.match("\[num\d+\]", w)
                if m is not None:
                    e = m.end()
                    v = int(w[:e][4:-1])
                    ids.append(self.ops_size + v)
                else:
                    if w not in self.words_index:
                        w = "[unk]"
                    ids.append(self.words_index[w])
            batch_ids.append(ids)
        target_ids = torch.tensor(batch_ids, dtype=torch.long)
        target_ids[target_ids == self.pad_token_id] = -100
        
        batch_ids = [[self.bos_token_id] + ids[:-1] for ids in batch_ids]
        decoder_input_ids = torch.tensor(batch_ids, dtype=torch.long)
        return decoder_input_ids, target_ids

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[int, List[int]]:
        if isinstance(ids, int):
            if ids < len(self.ext_words):
                return self.ext_words[ids]
            else:
                x = ids - len(self.ext_words)
                return f"[num{x}]"
        else:
            return [self.convert_ids_to_tokens(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: List[Tok]) -> List[int]:
        ids = []
        for w in tokens:
            m: re.Match = re.match("\[num\d+\]", w)
            if m is not None:
                e = m.end()
                v = int(w[:e][4:-1])
                ids.append(self.ops_size + v)
            else:
                if w not in self.words_index:
                    w = "[unk]"
                ids.append(self.words_index[w])
        return ids


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


class SeqDecoder(nn.Module):

    def __init__(self, hidden_dim: int, ops_size: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ops_size = ops_size
        self.inf = 1e12

        self.W_ops0 = nn.parameter.Parameter(torch.randn(ops_size, hidden_dim))
        self.W_ops1 = nn.parameter.Parameter(torch.randn(ops_size, hidden_dim))

        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def embedding(self, input_ids: Tensor, W_nums: Tensor) -> Tensor:
        N = input_ids.shape[0]

        W_ops = self.W_ops0.repeat([N, 1, 1])
        W_voc = torch.cat((W_ops, W_nums), dim=1)  # [N, V, D]

        input_ids = input_ids.unsqueeze(dim=-1).repeat([1, 1, self.hidden_dim])  # [N, L, D]
        inputs = torch.gather(W_voc, dim=1, index=input_ids)
        return inputs
    
    def forward(
        self,
        decoder_input_ids: Tensor,  # [N] or [N, L]
        past_value: Tensor,  # [N, D],
        W_nums: Tensor,  # [N, nums_size, D],
        output_mask: Tensor,  # [N, voc_size]
        encoder_outputs: Tensor,  # [N, L', D]
        attention_mask: Tensor,  # [N, L']
    ) -> Tensor:
        dim_equal_one = (decoder_input_ids.dim() == 1)
        if dim_equal_one:
            decoder_input_ids = decoder_input_ids.unsqueeze(dim=-1)  # [N] -> [N, L]
        decoder_inputs = self.embedding(decoder_input_ids, W_nums)
        qs, hn = self.gru(decoder_inputs, past_value.unsqueeze(dim=0))  # queries [N, L, D], hn [1, N, D]
        cs = self.attention(qs, encoder_outputs, attention_mask)  # contexts [N, L, D]
        feat = torch.tanh(self.fc(torch.cat((qs, cs), dim=-1)))  # feature [N, L, D]

        N = W_nums.shape[0]
        wn = torch.cat((self.W_ops1.unsqueeze(0).repeat([N, 1, 1]), W_nums), dim=1).transpose(1, 2)  # [N, D, voc_size]
        logits = torch.bmm(feat, wn)  # [N, L, voc_size]
        logits = logits.masked_fill((~output_mask).unsqueeze(dim=1), -self.inf)
        if dim_equal_one:
            logits = logits.squeeze(dim=1)  # [N, voc_size]
        return logits, hn.squeeze(dim=0)


class BertEncoder(nn.Module):

    def __init__(self, bert_name: str) -> None:
        super().__init__()
        self.impl: BertModel = BertModel.from_pretrained(bert_name)
    
    def forward(
        self,
        input_dict: Dict[str, Tensor]
    ) -> Tensor:
        encoder_outputs = self.impl(**input_dict).last_hidden_state  # [N, L, D]
        return encoder_outputs


class Seq2seqSolver:
    
    def __init__(
        self,
        cfg_dict: Dict[AnyStr, Any],
        ext_words: List[str],
        is_prop: bool = False,
    ) -> None:
        self.cfg = Config(**cfg_dict)

        self.enc_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.cfg.model_name)
        self.dec_tokenizer = DecoderTokenizer(ext_words, self.cfg.max_nums_size)

        self.encoder = BertEncoder(self.cfg.model_name)
        self.decoder = SeqDecoder(self.cfg.model_hidden_size, self.dec_tokenizer.ops_size)
        
        self.num_tokens_ids: List[int] = None
        self.dense_index = None

        self.update_voc(is_prop)
        self.to_device(self.cfg.device)

    def save_model(self, dir_path: str, suffix: str = "") -> None:
        enc_path = os.path.join(dir_path, f"seq_enc_{suffix}.pth")
        torch.save(self.encoder.state_dict(), enc_path)
        dec_path = os.path.join(dir_path, f"seq_dec_{suffix}.pth")
        torch.save(self.decoder.state_dict(), dec_path)
         
    def load_model(self, dir_path: str, suffix: str = "") -> None:
        enc_path = os.path.join(dir_path, f"seq_enc_{suffix}.pth")
        self.encoder.load_state_dict(torch.load(enc_path))
        dec_path = os.path.join(dir_path, f"seq_dec_{suffix}.pth")
        self.decoder.load_state_dict(torch.load(dec_path))

    def to_device(self, device: str) -> None:
        self.cfg.device = device
        self.encoder.to(device)
        self.decoder.to(device)
    
    def train(self) -> None:
        self.encoder.train()
        self.decoder.train()
    
    def eval(self) -> None:
        self.encoder.eval()
        self.decoder.eval()

    def update_voc(self, is_prop: bool) -> None:
        if is_prop:
            new_tokens = ['[expr]', '[num]'] \
                + [f'[num{n}]' for n in range(self.cfg.max_nums_size)] \
                + ['[int]', '[float]', '[frac]', '[perc]'] \
                + [f'[rk{n}]' for n in range(self.cfg.max_nums_size)]
        else:
            new_tokens = ['[example:text]', '[example:expr]', '[num]'] \
                + [f'[num{n}]' for n in range(self.cfg.max_nums_size)] \
                + [f'[exa_num{n}]' for n in range(self.cfg.max_nums_size)] \
                + ['[int]', '[float]', '[frac]', '[perc]'] \
                + [f'[rk{n}]' for n in range(self.cfg.max_nums_size)] \
                + [f'[exa_rk{n}]' for n in range(self.cfg.max_nums_size)]
        self.enc_tokenizer.add_tokens(new_tokens)
        self.encoder.impl.resize_token_embeddings(len(self.enc_tokenizer))

        self.num_tokens_ids = self.enc_tokenizer.convert_tokens_to_ids(
            [f'[num{n}]' for n in range(self.cfg.max_nums_size)]
        )
    
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
    
    def prepare_output(
        self, 
        batch_tokens: List[List[Tok]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        decoder_input_ids, target_ids = self.dec_tokenizer.batch_tokenize(batch_tokens)
        return decoder_input_ids.to(self.cfg.device), target_ids.to(self.cfg.device)
    
    def encode(
        self, 
        input_dict: Dict[str, Tensor], 
        num_ids: List[List[int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ops_size = self.dec_tokenizer.ops_size
        nums_size = self.dec_tokenizer.num_size
        N = len(num_ids)
        encoder_outputs: Tensor = self.encoder(input_dict)
        W_nums = torch.zeros(N, self.cfg.max_nums_size, self.cfg.model_hidden_size, dtype=torch.float, device=encoder_outputs.device)
        output_mask = torch.zeros(N, ops_size + nums_size, dtype=torch.bool, device=encoder_outputs.device)
        for i in range(N):
            n = len(num_ids[i])
            W_nums[i, :n, :].copy_(encoder_outputs[i, num_ids[i], :])
            output_mask[i, :ops_size + n] = True
        text_embedding = encoder_outputs[:, 0, :].clone()
        return text_embedding, encoder_outputs, W_nums, output_mask

    def decode(
        self,
        decoder_input_ids: Tensor,  # [N] or [N, L]
        past_value: Tensor,  # [N, D],
        W_nums: Tensor,  # [N, nums_size, D],
        output_mask: Tensor,  # [N, voc_size]
        encoder_outputs: Tensor,  # [N, L', D]
        attention_mask: Tensor,  # [N, L']        
    ) -> Tuple[Tensor, Tensor]:
        return self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_value=past_value,
            W_nums=W_nums,
            output_mask=output_mask,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask
        )

    @torch.no_grad()
    def generate(self, tokens: List[Tok], max_length: int = 30) -> Union[str, Tuple[str, Dict]]:
        self.eval()

        input_dict, nums_ids = self.prepare_input([tokens])
        text_embedding, encoder_outputs, W_nums, output_mask = self.encode(input_dict, nums_ids)
        past_value = text_embedding
        decoder_input_ids = torch.tensor(
            self.dec_tokenizer.bos_token_id, 
            dtype=torch.long,
            device=past_value.device
        ).view(1, 1)

        predict_ids = []

        while len(predict_ids) < max_length:
            next_token_logits, past_value = self.decode(
                decoder_input_ids=decoder_input_ids[:, -1],
                past_value=past_value,
                W_nums=W_nums,
                output_mask=output_mask,
                encoder_outputs=encoder_outputs,
                attention_mask=input_dict["attention_mask"]
            )
            next_token_id = torch.argmax(next_token_logits, dim=1, keepdim=True)  # [1, 1]
            
            if next_token_id.item() == self.dec_tokenizer.eos_token_id:
                break
            else:
                predict_ids.append(next_token_id.item())
                decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)  # [1, L] -> [1, L + 1]
        
        predict_tokens = self.dec_tokenizer.convert_ids_to_tokens(predict_ids)
        predict_text = "".join(predict_tokens)
        return predict_text
