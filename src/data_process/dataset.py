from asyncio.log import logger
from copy import deepcopy
import json
import os
import re
import jieba
from typing import Any, AnyStr, Dict, List, Tuple
from tqdm import tqdm
from utils import build_Op_list, \
    build_OpSeq_list_v1, build_OpSeq_list_v2, build_OpSeq_list_v3, \
    convert_const_nums


def seg_and_tag(expr: str, nums: List[str], fracs: List[str], tag: List[bool]) -> List[str]:  # seg the equation and tag the num
    seg_expr = []
    for frac in fracs:
        if frac in expr:
            p_start = expr.find(frac)
            p_end = p_start + len(frac)
            if p_start > 0:
                seg_expr += seg_and_tag(expr[:p_start], nums, fracs, tag)
            if nums.count(frac) > 0:
                if nums.count(frac) > 1:
                    tag[0] = True
                seg_expr.append("[num{}]".format(nums.index(frac)))
            else:
                seg_expr.append(frac)
            if p_end < len(expr):
                seg_expr += seg_and_tag(expr[p_end:], nums, fracs, tag)
            return seg_expr
    num_pos = re.search("\d+\.\d+%?|\d+%?", expr)
    if num_pos:
        p_start = num_pos.start()
        p_end = num_pos.end()
        if p_start > 0:
            seg_expr += seg_and_tag(expr[:p_start], nums, fracs, tag)
        num = expr[p_start:p_end]
        if nums.count(num) > 0:
            if nums.count(num) > 1:
                tag[0] = True
            seg_expr.append("[num{}]".format(nums.index(num)))
        else:
            seg_expr.append(num)
        if p_end < len(expr):
            seg_expr += seg_and_tag(expr[p_end:], nums, fracs, tag)
        return seg_expr
    for c in expr:
        seg_expr.append(c)
    return seg_expr


def preprocess(data: List[Dict]) -> List[Dict]:
    pattern = re.compile("\(\d+/\d+\)|\d+\.\d+%?|\d+%?")
    obj_data = []
    for sample in data:
        expr: str = sample["equation"][2:]
        
        expr = expr.replace("[", "(").replace("]", ")")  # replace [] to ()

        seg_text = sample["segmented_text"].strip().split(" ")

        nums  = []
        fracs = []
        tag = [False]

        _seg_text = []
        counter = 0
        for s in seg_text:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                _seg_text.append("[num{}]".format(counter))
                counter += 1
                nums.append(s[pos.start(): pos.end()])
                if pos.end() < len(s):
                    _seg_text.append(s[pos.end():])
            else:
                _seg_text.append(s)

        for num in nums:
            if re.search("\(\d+/\d+\)", num):
                fracs.append(num)
        fracs = sorted(fracs, key=lambda x: len(x), reverse=True)

        seg_text = _seg_text
        seg_expr = seg_and_tag(expr, nums, fracs, tag)
        
        nums_type = []
        for num in nums:
            if "%" in num:
                nums_type.append('[prec]')
            elif re.match("\d+\.\d+", num):
                nums_type.append('[float]')
            elif re.match("\d+", num):
                nums_type.append('[int]')
            else:
                nums_type.append('[frac]')

        def eval_fn(x):
            if '%' in x:
                return eval(x[:-1]) * 0.01
            else:
                return eval(x)

        nums_sorted = sorted(nums, key=eval_fn)
        nums_rank = []
        for n in nums:
            idx = nums_sorted.index(n)
            nums_rank.append(f'[rk{idx}]')

        cnt = 0
        for i in range(len(seg_text)):
            if re.match('\[num\d+\]', seg_text[i]):
                seg_text[i] = seg_text[i] + nums_rank[cnt] + nums_type[cnt]
                cnt += 1
                
        obj_data.append({
            "sample_id": sample["id"],
            "raw_text": sample["original_text"],
            "seg_text": seg_text,
            "seg_expr": seg_expr,
            "nums": nums,
            "tag": tag[0]
        })
    return obj_data


def loadMath23K(data_path: str, fold: int = -1, head: int = None) -> Tuple[List[Dict], List[Dict]]:
    train_data = []
    test_data  = []

    if fold == -1:
        traindata_path = os.path.join(data_path, "math23k_train.json")
        testdata_path = os.path.join(data_path, "math23k_test.json")
        for data, path in zip([train_data, test_data], [traindata_path, testdata_path]):
            with open(path, "r") as f:
                js = ""
                for i, s in enumerate(f):
                    js += s
                    i += 1
                    if i % 7 == 0:  # every 7 line is a json
                        data_d = json.loads(js)
                        if "千米/小时" in data_d["equation"]:
                            data_d["equation"] = data_d["equation"][:-5]
                        data.append(data_d)
                        js = ""
        
        if head is not None and head != -1:
            train_data = train_data[:head]
            test_data = test_data[:head // 10]

        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    else:
        data_path = os.path.join(data_path, "math23k.json")
        data = []
        with open(data_path, "r") as f:
            js = ""
            for i, s in enumerate(f):
                js += s
                i += 1
                if i % 7 == 0:  # every 7 line is a json
                    data_d = json.loads(js)
                    if "千米/小时" in data_d["equation"]:
                        data_d["equation"] = data_d["equation"][:-5]
                    data.append(data_d)
                    js = ""
        
        if head is not None and head != -1:
            data = data[:head]
        
        data = preprocess(data)
        
        if head is not None:
            for idx in range(5):
                print(data[idx])
        
        # 划分数据集
        fold_size = int(len(data) * 0.2)
        fold_data    = []
        for split_fold in range(4):
            fold_start = fold_size * split_fold
            fold_end = fold_size * (split_fold + 1)
            fold_data.append(data[fold_start:fold_end])
        fold_data.append(data[(fold_size * 4):])
        
        for fold_idx in range(5):
            if fold_idx == fold:
                test_data += fold_data[fold_idx]
            else:
                train_data += fold_data[fold_idx]

    return train_data, test_data


def loadDAG(data_path: str, head: int = None) -> Tuple[List[Dict], List[Dict]]:
    train_data = []
    test_data  = []

    traindata_path = os.path.join(data_path, "train.json")
    testdata_path = os.path.join(data_path, "test.json")
    for data, path in zip([train_data, test_data], [traindata_path, testdata_path]):
        with open(path, "r") as f:
            data.extend(json.load(f))
    
    if head is not None and head != -1:
        train_data = train_data[:head]
        test_data = test_data[:head // 10]

    return train_data, test_data


def build_ext_words(dataset: List[Dict], threshold: int = 5) -> List[str]:
    ext_words: Dict[str, int] = {}
    for obj in dataset:
        seg_expr = obj["seg_expr"]
        for word in seg_expr:
            if re.match("\[num\d*\]", word):
                continue
            if word not in ext_words:
                ext_words[word] = 0
            ext_words[word] += 1
    ext_words = [k for k, v in ext_words.items() if v >= threshold]
    return ext_words


def join_const_nums(dataset: List[Dict], const_nums):
    filter_count = 0
    new_dataset = []
    for obj in dataset:
        try:
            obj["const_nums"] = deepcopy(const_nums)
            obj["seg_expr"] = convert_const_nums(obj["seg_expr"], const_nums)
            new_dataset.append(obj)
        except:
            filter_count += 1
    print("filter count:", filter_count)
    return new_dataset


def join_Op_list(dataset: List[Dict]):
    filter_count = 0
    for obj in dataset:
        try:
            obj["Op_list"] = build_Op_list(obj["seg_expr"], obj["nums"])
        except:
            filter_count += 1
    print("filter count:", filter_count)
    return filter_count


def join_OpSeq_list(dataset: List[Dict], mode: str):
    filter_count = 0
    new_dataset = []
    for obj in dataset:
        try:
            if mode == "v1":
                obj["OpSeq_list"] = build_OpSeq_list_v1(obj["seg_expr"], len(obj["nums"]))
            elif mode == "v2":
                obj["OpSeq_list"] = build_OpSeq_list_v2(obj["seg_expr"], len(obj["nums"]))
            elif mode == "v3":
                obj["OpSeq_list"] = build_OpSeq_list_v3(obj["seg_expr"], len(obj["nums"]))
            else:
                raise ValueError
            new_dataset.append(obj)
        except SyntaxError:
            print(obj["raw_text"], obj["seg_expr"], obj["nums"])
            filter_count += 1
    print("filter count: {}".format(filter_count))
    return new_dataset
