from loguru import logger
from copy import deepcopy
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

Op = None

Tok = str

Expr = namedtuple('Expr', ['arg0', 'expr_toks', 'expr_str'])

MultiExpr = namedtuple('MultiExpr', ['args', 'expr_toks', 'expr_str'])

class MathDataset(Dataset):
    
    def __init__(self, data: List[Dict]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        return self.data[index]   


class MathDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        expr_list: List[Expr],
        target: Optional[List[Expr]] = None,
        id: Optional[int] = None,
        end: bool = True
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.expr_list = expr_list
        self.target = target
        self.id = id
        self.end = end
    
    def parse_input(self, sep_token: str, use_expr: bool = True) -> str:
        input_text = [self.question]
        if use_expr:
            for expr in self.expr_list:
                input_text.append(sep_token)
                input_text.append("[num{}] = {}"\
                    .format(expr.arg0, expr.expr_str))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self, bos_token: str, eos_token: str) -> str:
        output_text = []
        for expr in self.target:
            output_text.extend(expr.expr_toks)
            output_text.append(bos_token)
        if self.end:
            output_text[-1] = eos_token
        output_text = " ".join(output_text)
        return output_text


class TemplateDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        expr_list: List[MultiExpr],
        target: Optional[List[MultiExpr]] = None,
        id: Optional[int] = None,
        end: bool = True
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.expr_list = expr_list
        self.target = target
        self.id = id
        self.end = end
    
    def parse_input(self, sep_token: str, use_expr: bool = True) -> str:
        input_text = [self.question]
        if use_expr:
            for expr in self.expr_list:
                input_text.append(sep_token)
                input_text.append("{} = {}"\
                    .format(" ".join([f"[num{i}]" for i in expr.args]), " ".join(expr.expr_toks)))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self, bos_token: str, eos_token: str) -> str:
        output_text = []
        for expr in self.target:
            output_text.extend(expr.expr_toks)
            output_text.append(bos_token)
        if self.end:
            output_text[-1] = eos_token
        output_text = " ".join(output_text)
        return output_text


def convert_const_nums(seg_expr: List[Tok], const_nums: List[str]) -> int:
    new_seg_expr: List[str] = []
    for tok in seg_expr:
        if tok in "+*/^()" or re.match("\[num\d+\]", tok):
            new_seg_expr.append(tok)
        elif tok == '-':
            if len(new_seg_expr) > 0 and \
                (re.match("\[num\d+\]", new_seg_expr[-1]) \
                    or re.match("\[c\d+\]", new_seg_expr[-1]) \
                    or new_seg_expr[-1] == ')'):
                new_seg_expr.append(tok)
            else:
                idx = const_nums.index('-1')
                new_seg_expr.append(f"[c{idx}]")
                new_seg_expr.append("*")
        else:
            if tok not in const_nums:
                print("tok:", tok)
                print("const_nums:", const_nums)
                raise ValueError
            idx = const_nums.index(tok)
            new_seg_expr.append(f"[c{idx}]")
    return new_seg_expr


def seq2seq_parse_num_index(num_token: str, nums: List[str]) -> int:
    m = re.match("\[num(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        if num_token not in nums:
            raise ValueError
        return nums.index(num_token)


def parse_num_index(num_token: str) -> int:
    m = re.match("\[num(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        raise ValueError


def parse_const_num_index(num_token: str) -> int:
    m = re.match("\[c(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        raise ValueError


def parse_value(x: str) -> float:
    x = x.replace("%","*0.01")
    try:
        value = Decimal(eval(x))
        return value
    except:
        print(x)
        exit(-1)


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
            idx = seq2seq_parse_num_index(expr[:token_length], nums)
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


def build_Expr_list_v1(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 根据括号对表达式进行切分
    Expr_list: List[Expr]   = []
    match_pos: Dict[int, int] = {}
    expr_dict: Dict[str, int] = {f'[num{i}]': i for i in range(nums_size)}

    def compute_match_pos():
        stk = []
        for i, tok in enumerate(seg_expr):
            if tok == '(':
                stk.append(i)
            elif tok == ')':
                j = stk.pop()
                match_pos[j] = i

    def rec_build_expr_list(l: int, r: int):
        expr_toks = []
        i = l
        while i < r:
            tok = seg_expr[i]
            if tok.replace(" ", "") != "":
                if tok == '(':
                    arg = rec_build_expr_list(i + 1, match_pos[i])
                    expr_toks.append('[num{}]'.format(arg))
                    i = match_pos[i]
                else:
                    expr_toks.append(tok)
            i += 1
        expr_str = " ".join(expr_toks)
        if expr_str not in expr_dict:
            arg0 = len(expr_dict)
            expr_dict[expr_str] = arg0
            Expr_list.append(
                Expr(arg0=arg0, expr_toks=expr_toks, expr_str=expr_str)
            )
        return expr_dict[expr_str]
    
    compute_match_pos()
    rec_build_expr_list(0, len(seg_expr))

    return Expr_list


def build_Expr_list_v2(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 不对表达式进行切分
    expr_toks = seg_expr
    Expr_list: List[Expr] = [Expr(arg0=nums_size, expr_toks=expr_toks, expr_str=" ".join(expr_toks))]
    return Expr_list


def build_Expr_list_v3(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 根据运算符对表达式进行切分
    if len(seg_expr) == 1:
        return [Expr(arg0=nums_size, expr_toks=seg_expr, expr_str=" ".join(seg_expr))]

    Expr_list: List[Expr] = []
    expr_dict: Dict[str, int] = {f'[num{i}]': f'[num{i}]' for i in range(nums_size)}

    op_stack = []
    v_stack = []

    def pop_stack():
        op = op_stack.pop()
        arg2 = v_stack.pop()
        arg1 = v_stack.pop()
        expr_toks=[arg1, op, arg2]
        expr_str=f'{arg1} {op} {arg2}'
        if expr_str not in expr_dict:
            arg0 = len(expr_dict)
            expr_dict[expr_str] = f'[num{arg0}]'
            Expr_list.append(Expr(
                arg0=arg0, 
                expr_toks=expr_toks,
                expr_str=expr_str
            ))
        v_stack.append(expr_dict[expr_str])
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
            v_stack.append(t)
    while len(op_stack) > 0:
        pop_stack()
    if len(v_stack) != 1:
        raise SyntaxError

    return Expr_list


def compute_Expr_list(Expr_list: List[Expr], nums: List[str], const_nums: List[str], max_nums_size: int):
    if Expr_list is None:
        return None
    
    nums = [parse_value(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))

    const_nums = [parse_value(x) for x in const_nums]

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
            if t == '(':
                op_stack.append('(')
            elif t == ')':
                while len(op_stack) > 0 and op_stack[-1] != '(':
                    pop_stack()
                if len(op_stack) == 0:
                    logger.warning("decimal.Error: {}".format(Expr_list))
                    return None
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
                if re.match('\[num\d+\]', t):
                    i = parse_num_index(t)
                    v_stack.append(Decimal(nums_table[i]))
                else:
                    i = parse_const_num_index(t)
                    v_stack.append(Decimal(const_nums[i]))
            
        while len(op_stack) > 0:
            pop_stack()
        if len(v_stack) != 1:
            raise SyntaxError
        return v_stack[-1]

    try:
        for opSeq in Expr_list:
            nums_table[opSeq.arg0] = do_OpSeq(opSeq.expr_toks)
    except:
        logger.warning("decimal.Error: {}".format(Expr_list))
        return None
    return nums_table[Expr_list[-1].arg0] if len(Expr_list) > 0 else None


def compute_MultiExpr_list(MultiExpr_list: List[MultiExpr], nums: List[str], const_nums: List[str], max_nums_size: int):
    if MultiExpr_list is None:
        return None
    
    nums = [float(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))

    const_nums = [float(x) for x in const_nums]

    def parse_quants(tokens: List[Tok]):
        return [parse_num_index(token) for token in tokens]

    def parse_fn(tokens: List[Tok]):
        # print("op:", "".join(tokens))
        # print("tokens:", tokens)
        quantsIndex = parse_quants(tokens[1:])
        if tokens[0] == "[solve_linear_equation]":
            A = np.array(
                [
                    [nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]],
                    [nums_table[quantsIndex[2]], nums_table[quantsIndex[3]]],
                ]
            )
            b = np.array(
                [nums_table[quantsIndex[4]], nums_table[quantsIndex[5]]]
            )
            return (np.linalg.inv(A) @ b).tolist()
        elif tokens[0] == "[quadratic_function_integral]":
            def poly3(a, b, c, d, x):
                return a * (x ** 3) + b * (x ** 2) + c * x + d
            l, r = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            a0, a1, a2 = nums_table[quantsIndex[2]], nums_table[quantsIndex[3]], nums_table[quantsIndex[4]]
            return [poly3(a0 / 3, a1 / 2, a2, 0, r) - poly3(a0 / 3, a1 / 2, a2, 0, l)]
        elif tokens[0] == "[quadratic_function_extremum]":
            def poly2(a, b, c, x):
                return a * (x ** 2) + b * x + c
            a0, a1, a2 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]], nums_table[quantsIndex[2]]
            return [poly2(a0, a1, a2, -a1 / (2 * a0))]
        elif tokens[0] == "[add]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 + a1]
        elif tokens[0] == "[sub]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 + a1]
        elif tokens[0] == "[mul]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 * a1]
        elif tokens[0] == "[div]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 / a1]
        else:
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 ** a1]

    def do_OpSeq(expr_toks: List[Tok]):
        return parse_fn(expr_toks)

    try:
        for expr in MultiExpr_list:
            results = do_OpSeq(expr.expr_toks)
            for i, index in enumerate(expr.args):
                nums_table[index] = results[i]
    except:
        logger.warning("decimal.Error: {}".format(MultiExpr_list))
        return None
    return nums_table[MultiExpr_list[-1].args[-1]] if len(MultiExpr_list) > 0 else None

"""
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
            raise SyntaxError
    try:
        for Op in Op_list:
            nums_table[Op.arg0] = do_Op(Op.arg1, Op.arg2, Op.op)
    except:
        logger.warning("decimal.Error: {}".format(Op_list))
        return None
    return nums_table[Op_list[-1].arg0] if len(Op_list) > 0 else None
"""