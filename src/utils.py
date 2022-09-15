from loguru import logger
import numpy as np
import torch
import decimal
from decimal import Decimal
from torch import Tensor
from torch.utils.data import Dataset

import re
import random
from collections import namedtuple
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union

Tok = str

Op = namedtuple('Op', ['arg0', 'arg1', 'arg2', 'op'])  # $(arg0) = $(arg1) $(op) $(arg2)

OpSeq = namedtuple('OpList', ['arg0', 'expr_toks', 'expr_str'])

class DefaultDataset(Dataset):
    
    def __init__(self, data: List[Dict]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        return self.data[index]    


class OpDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        op_list: List[Op],
        expr_list: List[str],
        target: Op
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.op_list = op_list
        self.expr_list = expr_list
        self.target = target
    
    def parse_input(self) -> str:
        input_text = [self.question]
        for i in range(len(self.nums) - len(self.const_nums), len(self.nums)):
            input_text.append("[num{}] = {}.".format(i, self.nums[i]))
        for op in self.op_list:
            input_text.append("[SEP]")
            # input_text.append("[num{}] = {}"\
            #     .format(op.arg0, self.expr_list[op.arg0]))
            input_text.append("[num{}] = [num{}] {} [num{}]"\
                .format(op.arg0, op.arg1, op.op, op.arg2))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self) -> str:
        if self.target.op != "=":
            op = self.target
            return "[num{}] {} [num{}]".format(op.arg1, op.op, op.arg2)
        else:
            return "= [num{}]".format(self.target.arg1)


class OpSeqDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        opSeq_list: List[OpSeq],
        target: Optional[List[OpSeq]] = None,
        id: Optional[int] = None
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.opSeq_list = opSeq_list
        self.target = target
        self.id = id
    
    def parse_input(self) -> str:
        input_text = [self.question]
        for i in range(len(self.nums) - len(self.const_nums), len(self.nums)):
            input_text.append("[num{}] = {}.".format(i, self.nums[i]))
        for opSeq in self.opSeq_list:
            input_text.append("[SEP]")
            input_text.append("[num{}] = {}"\
                .format(opSeq.arg0, opSeq.expr_str))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self) -> str:
        output_text = []
        for i, opSeq in enumerate(self.target):
            output_text.append(opSeq.expr_str)
            if i + 1 != len(self.target):
                output_text.append("[eos0]")
            else:
                output_text.append("[eos1]")
        output_text = " ".join(output_text)
        return output_text


def parse_num_index(num_token: str, nums: List[str]) -> int:
    m = re.match("\[num(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        if num_token not in nums:
            raise ValueError
        return nums.index(num_token)

def parse_value(x: str) -> float:
    x = x.replace("%","*0.01")
    try:
        value = Decimal(eval(x))
        return value
    except:
        print(x)
        exit(-1)


def build_Op_list(seg_expr: List[Tok], nums: List[str]) -> List[Op]:
    if len(seg_expr) == 1:
        return build_Op_list(seg_expr + ['*', '1'], nums)
    nums_size = [len(nums)]
    Op_list = []

    op_stack = []
    v_stack = []

    def pop_stack():        
        op = op_stack.pop()
        arg2 = v_stack.pop()
        arg1 = v_stack.pop()
        arg0 = nums_size[0]
        Op_list.append(Op(arg0=arg0, arg1=arg1, arg2=arg2, op=op))
        v_stack.append(arg0)
        nums_size[0] += 1
    
    for t in seg_expr:
        if t.replace(" ", "") == "":
            continue
        if t == '(':
            op_stack.append('(')
        elif t == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                pop_stack()
            op_stack.pop()
        elif t == '+' or t == '-':
            while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '*' or t == '/':
            while len(op_stack) > 0 and op_stack[-1] in '*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '^':
            while len(op_stack) > 0 and op_stack[-1] in '^':
                pop_stack()
            op_stack.append(t)
        else:
            v_stack.append(parse_num_index(t, nums))
    while len(op_stack) > 0:
        pop_stack()
    if len(v_stack) != 1:
        raise SyntaxError
    return Op_list


def compute_Op_list(Op_list: List[Op], nums: List[str], max_nums_size: int) -> float:

    nums = [parse_value(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))
 
    def do_Op(arg1, arg2, op):
        if op == '+':
            return nums_table[arg1] + nums_table[arg2]
        elif op == '-':
            return nums_table[arg1] - nums_table[arg2]
        elif op == '*':
            return nums_table[arg1] * nums_table[arg2]
        elif op == '/':
            return nums_table[arg1] / nums_table[arg2]
        elif op == '^':
            return pow(nums_table[arg1], nums_table[arg2])
        elif op == '=':
            return nums_table[arg1]
        else:
            raise ValueError
    try:
        for Op in Op_list:
            nums_table[Op.arg0] = do_Op(Op.arg1, Op.arg2, Op.op)
    except:
        logger.warning("decimal.Error: {}".format(Op_list))
        return None
    return nums_table[Op_list[-1].arg0] if len(Op_list) > 0 else None


def build_OpSeq_list_v1(seg_expr: List[Tok], nums: List[str], debug: bool = False) -> List[OpSeq]:
    OpSeq_list: List[OpSeq]   = []
    match_pos: Dict[int, int] = {}
    expr_dict: Dict[str, int] = {f'[num{i}]': i for i in range(len(nums))}

    def compute_match_pos():
        stk = []
        for i, tok in enumerate(seg_expr):
            if tok == '(':
                stk.append(i)
            elif tok == ')':
                j = stk.pop()
                match_pos[j] = i

    def rec_build_OpSeq_list(l: int, r: int):
        expr_toks = []
        i = l
        while i < r:
            tok = seg_expr[i]
            if tok.replace(" ", "") != "":
                if tok == '(':
                    arg = rec_build_OpSeq_list(i + 1, match_pos[i])
                    expr_toks.append('[num{}]'.format(arg))
                    i = match_pos[i]
                elif tok in '+-*/^':
                    expr_toks.append(tok)
                else:
                    expr_toks.append('[num{}]'.format(parse_num_index(tok, nums)))
            i += 1
        expr_str = "".join(expr_toks)
        if expr_str not in expr_dict:
            arg0 = len(expr_dict)
            if debug:
                print("arg0:", arg0)
                print("expr_dict:", expr_dict)
            expr_dict[expr_str] = arg0
            OpSeq_list.append(
                OpSeq(arg0=arg0, expr_toks=expr_toks, expr_str=expr_str)
            )
        return expr_dict[expr_str]
    
    compute_match_pos()
    rec_build_OpSeq_list(0, len(seg_expr))

    return OpSeq_list


def compute_OpSeq_list(OpSeq_list: List[OpSeq], nums: List[str], max_nums_size: int):

    nums = [parse_value(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))

    def do_OpSeq(expr_toks: List[Tok]):
        # print("expr_tokens:", expr_toks)

        op_stack = []
        v_stack = []

        def pop_stack():
            o = op_stack.pop()
            v1 = v_stack.pop()
            v0 = v_stack.pop()
            # print("do_op: [{} {} {}]".format(v0, o, v1))
            if o not in '+-*/^':
                raise SyntaxError
            if o == '^':
                v_stack.append(pow(v0, v1))
            elif o == '+':
                v_stack.append(v0 + v1)
            elif o == '-':
                v_stack.append(v0 - v1)
            elif o == '*':
                v_stack.append(v0 * v1)
            elif o == '/':
                v_stack.append(v0 / v1)

        for t in expr_toks:
            if t.replace(" ", "") == "":
                continue
            if t == '+' or t == '-':
                while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                    pop_stack()
                op_stack.append(t)
            elif t == '*' or t == '/':
                while len(op_stack) > 0 and op_stack[-1] in '*/^':
                    pop_stack()
                op_stack.append(t)
            elif t == '^':
                while len(op_stack) > 0 and op_stack[-1] in '^':
                    pop_stack()
                op_stack.append(t)
            else:
                i = parse_num_index(t, nums)
                v_stack.append(Decimal(nums_table[i]))
        
        while len(op_stack) > 0:
            pop_stack()
        if len(v_stack) != 1:
            raise SyntaxError
        return v_stack[-1]

    try:
        for opSeq in OpSeq_list:
            nums_table[opSeq.arg0] = do_OpSeq(opSeq.expr_toks)
    except SyntaxError:
        logger.warning("decimal.Error: {}".format(OpSeq_list))
        return None
    return nums_table[OpSeq_list[-1].arg0] if len(OpSeq_list) > 0 else None


def eval_expr(tokens: List[Tok]):
    op_stack = []
    v_stack = []

    def pop_stack():
        o = op_stack.pop()
        v1 = v_stack.pop()
        v0 = v_stack.pop()
        if o not in '+-*/^':
            raise SyntaxError
        if o == '^':
            v_stack.append(pow(v0, v1))
        elif o == '+':
            v_stack.append(v0 + v1)
        elif o == '-':
            v_stack.append(v0 - v1)
        elif o == '*':
            v_stack.append(v0 * v1)
        elif o == '/':
            v_stack.append(v0 / v1)

    for t in tokens:
        if t.replace(" ", "") == "":
            continue
        if t == '(':
            op_stack.append('(')
        elif t == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                pop_stack()
            op_stack.pop()
        elif t == '+' or t == '-':
            while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '*' or t == '/':
            while len(op_stack) > 0 and op_stack[-1] in '*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '^':
            while len(op_stack) > 0 and op_stack[-1] in '^':
                pop_stack()
            op_stack.append(t)
        else:
            v_stack.append(Decimal(eval(t)))
    while len(op_stack) > 0:
        pop_stack()
    if len(v_stack) != 1:
        raise SyntaxError
    return v_stack[-1]


def convert_expr(expr: str, nums: List[str]):
    tokens = []
    while len(expr) > 0:
        m: re.Match = re.match("\[num\d+\]", expr)
        token_length = 0
        if m is None:
            token_length = 1
            tokens.append(expr[0])
        else:
            token_length = m.end()
            idx = parse_num_index(expr[:token_length], nums)
            num = nums[idx] if idx < len(nums) else '1'
            tokens.append("(" + num + ")")
        expr = expr[token_length:]
    expr = "".join(tokens)
    return expr


def compute_expr(expr: str, nums: List[str]):
    expr = convert_expr(expr, nums)
    expr = expr.replace("%", "*0.01")
    tokens = re.split(r"([\*\/\^\(\)\+\-])", expr)
    # print("".join(tokens))
    try:
        value = eval_expr(tokens)
    except:
        print("".join(tokens))
        value = None
    return value
