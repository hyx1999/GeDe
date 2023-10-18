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

const_nums = [1]

q_pat = re.compile('\[q(\d+)\]')
o_pat = re.compile('\[o(\d+)\]')
n_pat = re.compile('\[n(\d+)\]')

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
            "text": "Determine [o0] as the definite integral of quadratic function [q0] * x^2 + [q1] * x + [q2] between the intervals [q3] and [q4] .",
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
    
    # shuffle sentence
    # extSegText = " ".join(extSegText)
    # extSegText = extSegText.split(".")[:-1]
    # random.shuffle(extSegText)
    # extSegText = ".".join(extSegText).split() + ["."]
    
    segText = segText + extSegText + " Output the value of [q{}] .".format(N_Q - 1).split()
    
    # avg_len.append(sum(len(x["expr_toks"]) for x in exprList))
    avg_len.append(len(exprList))
    
    return {
        "sample_id": data_id,
        "raw_text": rawText,
        "seg_text": segText,
        "Expr_list": exprList,
        "const_nums": [],
        "nums": qs
    }

if __name__ == '__main__':
    setup_seend()
    folder_path = "../data/MathTemplate"

    train_path = os.path.join(folder_path, "train.json")
    dev_path = os.path.join(folder_path, "dev.json")
    test_path = os.path.join(folder_path, "test.json")

    for path, num in zip([train_path, dev_path, test_path], [1000, 100, 100]):
        data = [genDataInstance(str(_)) for _ in tqdm(range(num), total=num)]
        # with open(path, "w") as f:
        #     f.write(json.dumps(data, ensure_ascii=False))

    print(len([x for x in avg_len if x >= 4]))
    print(sum(avg_len) / len(avg_len))
    print(max(avg_len))
