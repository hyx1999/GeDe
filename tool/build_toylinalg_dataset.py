from collections import namedtuple
import random
import os
import json
import re
from typing import Dict, List
import numpy as np
from scipy import linalg
from tqdm import tqdm

np.random.seed(0)
random.seed(0) 

Transform = namedtuple("Transform", ["indices0", "mat0", "inv0", "indices1", "mat1", "inv1"])

def rand_mat(n: int):
    return np.random.rand(n, n) + 0.5

def rand_vec(n: int):
    v = np.random.rand(n) + 0.5
    return v

def to_str(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    else:
        return x

def mat_to_toks(m: np.ndarray, quant_map: Dict[str, str]) -> List[str]:
    n = m.shape[0]
    res = []
    for i in range(n):
        res.extend(
            ["["] + \
            ["[num{}]".format(quant_map["{:.3f}".format(m[i, j])]) for j in range(n)] + \
            ["]"]
        )
    return ["["] + res + ["]"]

def shuffle(x: List) -> List:
    x_copy = [i for i in x]
    random.shuffle(x_copy)
    return x_copy

def genDAGInstance(id: str):
    N = random.randint(1, 2)
    M = random.randint(1, 2)
    level = random.randint(2, 3)
    init_num = rand_vec(N + M)
    transform_per_level = []
    for i in range(1, level):
        indices0 = sorted(np.random.choice(N + M, N, replace=False).tolist())
        indices1 = sorted(np.array([i for i in range(N + M) if i not in indices0]).tolist())
        mat0 = rand_mat(N)
        mat1 = rand_mat(M)
        inv0 = random.randint(0, 1)
        inv1 = random.randint(0, 1)
        
        transform_per_level.append(
            Transform(
                mat0=mat0,
                indices0=indices0,
                inv0=inv0,
                mat1=mat1,
                indices1=indices1,
                inv1=inv1
            )
        )

    texts = []
    texts.append(" ".join(
        ["Given"] + \
        ["a_{} = {:.3f} ,".format(i, init_num[i]) for i in range(N + M)] + \
        [f"S_0 is the sum of [a_0, ..., a_{N+M-1}] ."])
    )
    for i in range(1, level):
        T: Transform = transform_per_level[i - 1]
        prev_index = (i - 1) * (N + M)
        index = i * (N + M)
        
        texts_i = []
        for x in range(N):
            if T.inv0:
                texts_i.append(
                    f"a_{prev_index + T.indices0[x]} = " + \
                    " + ".join(shuffle([f"{T.mat0[x, y]:.3f} * b_{index + y}" for y in range(N)])) + " , "
                )
            else:
                texts_i.append(
                    f"b_{index + x} = " + \
                    " + ".join(shuffle([f"{T.mat0[x, y]:.3f} * a_{prev_index + T.indices0[y]}" for y in range(N)])) + " , "
                )

        for x in range(M):
            if T.inv1:
                texts_i.append(
                    f"a_{prev_index + T.indices1[x]} = " + \
                    " + ".join(shuffle([f"{T.mat1[x, y]:.3f} * b_{index + N + y}" for y in range(M)])) + " , "
                )
            else:
                texts_i.append(
                    f"b_{index + N + x} = " + \
                    " + ".join(shuffle([f"{T.mat1[x, y]:.3f} * a_{prev_index + T.indices1[y]}" for y in range(M)])) + " , "
                )
        texts_i = shuffle(texts_i)
        texts_i.append(f"[a_{index}, ..., a_{index+N+M-1}] equal to [b_{index}, ..., b_{index+N+M-1}] multiply S_{i-1} ,")
        texts_i.append(f"S_{i} is the sum of [a_{index}, ..., a_{index+N+M-1}] .")
        texts.append(" ".join(texts_i))
    texts.append(f"find the value of S_{level - 1} .")
    
    raw_text = " ".join(texts)
        
    quant_map = {}
    nums = []
    new_texts = []
    for token in raw_text.split():
        if not re.fullmatch("[-]{0,1}\d+\.\d+", token):
            new_texts.append(token)
            continue
        token = "{:.3f}".format(float(token))
        if token not in quant_map:
            quant_map[token] = len(quant_map)
            nums.append(token)
        token = "[num{}]".format(quant_map[token])
        new_texts.append(token)
    text = " ".join(new_texts)
    
    expr_list = []
    quant_offset = len(quant_map)

    var_map = {}
    for x in range(N + M):
        var_map[f"a_{x}"] = f"[num{x}]"

    expr_list.append({
        "args": [quant_offset],
        "expr_toks": ["Sum", "[", "["] + [f"[num{x}]" for x in range(N + M)] + ["]", "]"]
    })
    var_map[f"S_0"] = f"[num{quant_offset}]"
    quant_offset += len(expr_list[-1]["args"])

    for i in range(1, level):
        index = i * (N + M)
        prev_index = (i - 1) * (N + M)
        T: Transform = transform_per_level[i - 1]
        fn0: str = "MatSolve" if T.inv0 else "MatMul"
        fn1: str = "MatSolve" if T.inv1 else "MatMul"
        expr_list.append({
            "args": [quant_offset + x for x in range(N)],
            "expr_toks": \
                [fn0] + ["["] + \
                mat_to_toks(T.mat0, quant_map) + \
                [","] + \
                ["["] + [var_map["a_{}".format(prev_index + T.indices0[x])] for x in range(N)] + ["]"] + \
                ["]"]
        })
        for x in range(N):
            var_map[f"b_{index + x}"] = f"[num{quant_offset + x}]"
        quant_offset += len(expr_list[-1]["args"])

        expr_list.append({
            "args": [quant_offset + x for x in range(M)],
            "expr_toks": \
                [fn1] + ["["] + \
                mat_to_toks(T.mat1, quant_map) + \
                [","] + \
                ["["] + [var_map["a_{}".format(prev_index + T.indices1[x])] for x in range(M)] + ["]"] + \
                ["]"]
        })
        for x in range(M):
            var_map[f"b_{index + N + x}"] = f"[num{quant_offset + x}]"
        quant_offset += len(expr_list[-1]["args"])
        
        expr_list.append({
            "args": [quant_offset + x for x in range(N + M)],
            "expr_toks": \
                ["Mul"] + ["["] + \
                ["["] + [var_map["b_{}".format(index + x)] for x in range(N + M)] + ["]"] + \
                [","] + \
                [var_map[f"S_{i - 1}"]] + \
                ["]"]
        })
        for x in range(N + M):
            var_map[f"a_{index + x}"] = f"[num{quant_offset + x}]"
        quant_offset += len(expr_list[-1]["args"])

        expr_list.append({
            "args": [quant_offset],
            "expr_toks": ["Sum", "[", "["] + [var_map["a_{}".format(index + x)] for x in range(N + M)] + ["]", "]"]
        })
        var_map[f"S_{i}"] = f"[num{quant_offset}]"
        quant_offset += len(expr_list[-1]["args"])
    
    return {
        "sample_id": id,
        "raw_text": raw_text,
        "seg_text": text,
        "nums": nums,
        "const_nums": [],
        'Expr_list': expr_list
    }

folder_path = "../data/ToyLinalg"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

train_path = os.path.join(folder_path, "train.json")
dev_path = os.path.join(folder_path, "dev.json")
test_path = os.path.join(folder_path, "test.json")

# obj = genDAGInstance('test')
# print(json.dumps(obj, indent=4))
# exit(0)

for path, num in zip([train_path, dev_path, test_path], [500, 100, 100]):
    data = [genDAGInstance(str(_)) for _ in tqdm(range(num), total=num)]
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False))
