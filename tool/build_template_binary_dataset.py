import random
import string
import os
import json
import re
import numpy as np
from typing import Dict, List, Set
from collections import namedtuple
from scipy import linalg
from tqdm import tqdm

avg_len = []

const_nums = [1, 2, 3]

q_pat = re.compile('\[q(\d+)\]')
o_pat = re.compile('\[o(\d+)\]')
n_pat = re.compile('\[n(\d+)\]')
num_pat = re.compile('\[num(\d+)\]')

templates = {
    "linalg": [
        {
            "text": "[q0] * [o0] + [q1] * [o1] = [q4] . [q2] * [o0] + [q3] * [o1] = [q5] .",
            "op": "[solve_linear_equation] [q0] [q1] [q2] [q3] [q4] [q5]",
            "in_num": 6,
            "out_num": 2,
        },
        {
            "text": "Determine [o0], [o1] as the result of inverse of matrix [ [ [q0] , [q1] ] , [ [q2] , [q3] ] ] times vector [ [q4] , [q5] ] .",
            "op": "[solve_linear_equation] [q0] [q1] [q2] [q3] [q4] [q5]",
            "in_num": 6,
            "out_num": 2,            
        }
    ],
    "integral": [
        {
            "text": "Determine [o0] as the definite integral of quadratic function [q0] * x^2 + [q1] * x + [q2] in the range [[q3] to [[q4]].",
            "op": "[quadratic_function_integral] [q3] [q4] [q0] [q1] [q2]",
            "in_num": 5,
            "out_num": 1,
        }
    ],
    "extremum": [
        {
            "text": "Determine [o0] as the the extremum value of quadratic function [q0] * x^2 + [q1] * x + [q2] .",
            "op": "[quadratic_function_extremum] [q0] [q1] [q2]",
            "in_num": 3,
            "out_num": 1,
        }
    ],
    "add": [
        {
            "text": "Determine [o0] as the sum of [q0] and [q1] .",
            "op": "[add] [q0] [q1]",
            "in_num": 2,
            "out_num": 1,
        }
    ],
    "sub": [
        {
            "text": "Determine [o0] as the [q0] minus [q1] .",
            "op": "[sub] [q0] [q1]",
            "in_num": 2,
            "out_num": 1,
        }
    ],
    "mul": [
        {
            "text": "Determine [o0] as the [q0] times [q1] .",
            "op": "[mul] [q0] [q1]",
            "in_num": 2,
            "out_num": 1,
        }
    ],
    "div": [
        {
            "text": "Determine [o0] as the [q0] divided by [q1] .",
            "op": "[div] [q0] [q1]",
            "in_num": 2,
            "out_num": 1,
        }
    ],
    "pow": [
        {
            "text": "Determine [o0] as the [q1] power of [q0] .",
            "op": "[pow] [q0] [q1]",
            "in_num": 2,
            "out_num": 1,
        }
    ],
}

def setup_seend():
    random.seed(0)
    np.random.seed(0)

def rand_f32():
    return random.random()

def rand_i32(a: int, b: int) -> int:
    return random.randint(a, b)

def rand_name(nameSet: Set[str]):
    while True:
        length = rand_i32(1, 6)
        letters = string.ascii_lowercase
        name = '(' + ''.join(random.choice(letters) for i in range(length)) + ')'
        if name not in nameSet:
            nameSet.add(name)
            return name

def genDataInstance(data_id: int):
    MAX_LEN = 450
    N   = 6
    N_O = rand_i32(1, 5)

    nameSet = set()
    segText    = []
    extSegText = []
    exprList   = []

    qs = [rand_f32() for _ in range(N)]
    Os = []
    
    for _ in range(N_O):
        opType = random.choice(list(templates.keys()))
        o = random.choice(templates[opType])
        Os.append(o)
    
    # build text
    N_Q = N
    ws = [1.0 for _ in range(N_Q)]
    segText.extend(
        ("Given " + " . ".join([f"[q{i}] = [num{i}]" for i in range(N)])).split() + [" . "]
    )
    for i in range(N_O):
        O = Os[i]
        in_indices: List[int] = np.random.choice(N_Q - 1, O["in_num"] - 1, replace=False).tolist() + [N_Q - 1]
        curSegText: List[str] = O["text"].split()
        curSegOp: List[str] = O["op"].split()
        # names: List[str] = [rand_name(nameSet) for _ in range(O["n_num"])]
        # # replace name
        # for i in range(len(curSegText)):
        #     m = n_pat.match(curSegText[i])
        #     if m:
        #         curSegText[i] = names[int(m.group(1))]
        # replace q
        for i in range(len(curSegText)):
            m = q_pat.match(curSegText[i])
            if m:
                idx = int(m.group(1))
                curSegText[i] = f"[q{in_indices[idx]}]"
        for i in range(len(curSegOp)):
            m = q_pat.match(curSegOp[i])
            if m:
                idx = int(m.group(1))
                curSegOp[i] = f"[num{in_indices[idx]}]"
        # replace o
        for i in range(len(curSegText)):
            m = o_pat.match(curSegText[i])
            if m:
                idx = int(m.group(1))
                curSegText[i] = f"[q{N_Q + idx}]"
        
        if len(segText) + len(extSegText) + len(curSegText) + sum(len(e["expr_toks"]) for e in exprList) < MAX_LEN:
            extSegText += curSegText       
            exprList.append({
                "args": [N_Q + idx for idx in range(O["out_num"])],
                "expr_toks": curSegOp
            })
            ws = [0.8 * x for x in ws] + [1.0] * O["out_num"]
            N_Q += O["out_num"]
    
    rawText = " ".join(segText + extSegText) + " Output the value of [q{}] .".format(N_Q - 1)    
    segText = segText + extSegText + " Output the value of [q{}] .".format(N_Q - 1).split()
    
    # translate exprList to one expression
    tree = namedtuple('tree', ['value', 'left', 'right'])
    trees = [tree(value=f'[num{i}]', left=None, right=None) for i in range(N)]
    
    def add(a, b):
        return tree('+', a, b)
    def sub(a, b):
        return tree('-', a, b)
    def mul(a, b):
        return tree('*', a, b)
    def div(a, b):
        return tree('/', a, b)
    def pow(a, b):
        return tree('^', a, b)
    C = {
        1: tree('[c0]', None, None),
        2: tree('[c1]', None, None),
        3: tree('[c2]', None, None),
        -1: tree('[c3]', None, None)
    }
    
    
    for expr in exprList:
        parseIndex = lambda x: int(num_pat.match(x).group(1))
        indices = [parseIndex(x) for x in expr["expr_toks"][1:]]
        if expr["expr_toks"][0] == "[solve_linear_equation]":
            a = trees[indices[0]]
            b = trees[indices[1]]
            c = trees[indices[2]]
            d = trees[indices[3]]
            x = trees[indices[4]]
            y = trees[indices[5]]
            det = sub(mul(a, d), mul(c, b))
            t_o0 = div(sub(mul(d,x),mul(b,y)),det)
            t_o1 = div(sub(mul(a,x),mul(c,y)),det)
            trees.append(t_o0)
            trees.append(t_o1)
        elif expr["expr_toks"][0] == "[quadratic_function_integral]":
            xl = trees[indices[0]]
            xr = trees[indices[1]]
            a = trees[indices[2]]
            b = trees[indices[3]]
            c = trees[indices[4]]
            xl3 = pow(xl, C[3])
            xl2 = pow(xl, C[2])
            xr3 = pow(xr, C[3])
            xr2 = pow(xr, C[2])
            vl = add(mul(div(a, C[3]), xl3), add(mul(div(b, C[2]), xl2), mul(c, xl)))
            vr = add(mul(div(a, C[3]), xr3), add(mul(div(b, C[2]), xr2), mul(c, xr)))
            t_o = sub(vr, vl)
            trees.append(t_o)
        elif expr["expr_toks"][0] == "[quadratic_function_extremum]":
            a = trees[indices[0]]
            b = trees[indices[1]]
            c = trees[indices[2]]
            x = mul(C[-1], div(b, mul(C[2], a)))
            x2 = pow(x, C[2])
            t_o = add(mul(a, x2), add(mul(b, x), c))
            trees.append(t_o)
        elif expr["expr_toks"][0] == "[add]":
            trees.append(tree('+', left=trees[indices[0]], right=trees[indices[1]]))
        elif expr["expr_toks"][0] == "[sub]":
            trees.append(tree('-', left=trees[indices[0]], right=trees[indices[1]]))
        elif expr["expr_toks"][0] == "[mul]":
            trees.append(tree('*', left=trees[indices[0]], right=trees[indices[1]]))
        elif expr["expr_toks"][0] == "[div]":
            trees.append(tree('/', left=trees[indices[0]], right=trees[indices[1]]))
        elif expr["expr_toks"][0] == "[pow]":
            trees.append(tree('^', left=trees[indices[0]], right=trees[indices[1]]))
    
    def genExpr(x: tree) -> list[str]:
        o = [x.value]
        if x.left:
            o.extend(genExpr(x.left))
        if x.right:
            o.extend(genExpr(x.right))
        return o
    expr = genExpr(trees[-1])[:256]
    avg_len.append(len(expr))
    
    return {
        "sample_id": data_id,
        "raw_text": rawText,
        "seg_text": segText,
        "Expr_list": [
            {
                "args": [N],
                "expr_toks": expr
            }    
        ],
        "const_nums": [1, 2, 3, -1],
        "nums": qs
    }

if __name__ == '__main__':
    setup_seend()
    folder_path = "../data/MathTemplateBinary"

    train_path = os.path.join(folder_path, "train.json")
    dev_path = os.path.join(folder_path, "dev.json")
    test_path = os.path.join(folder_path, "test.json")

    for path, num in zip([train_path, dev_path, test_path], [1000, 100, 100]):
        data = [genDataInstance(str(_)) for _ in tqdm(range(num), total=num)]
        with open(path, "w") as f:
            f.write(json.dumps(data, ensure_ascii=False))

    print(sum(avg_len) / len(avg_len))
    print(max(avg_len))