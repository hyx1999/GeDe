import numpy as np
import os
import json
from torch.utils.data import Dataset
from typing import Dict, AnyStr, List, Union, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict
from kbqa_utils import Expr, KBQADataset, RawDataInstance


def loadWebQSP(path: AnyStr, head: int) -> Dict[AnyStr, KBQADataset]:
    dataset_dict: Dict[str, KBQADataset] = dict()
    rel_dict = defaultdict(set)
    for key in ["train", "test"]:
        data_path = os.path.join(path, "data", "webqsp_0107.{}.json".format(key))
        data = []
        filter_count = 0
        with open(data_path, "r") as f:
            obj = json.loads(f.read())
            for raw_item in obj:
                qid = raw_item["qid"]
                query = raw_item["question"]
                S_expr = raw_item["s_expression_processed"]
                answer = [x["answer_argument"] for x in raw_item["answer"]]
                if S_expr is None:
                    filter_count += 1
                    continue
                item = RawDataInstance(qid, query, S_expr, answer)
                if key == "train" and not item.check_exprs():
                    filter_count += 1
                    continue
                data.append(item)               
                for edge in raw_item["graph_query"]["edges"]:
                    rel_dict[key].add(edge["relation"])

        if key == "train":
            dev_indexs = np.random.choice(len(data), 100, replace=False).tolist()
            train_indexs = [i for i in range(len(data)) if i not in dev_indexs]

            train_data = [data[i] for i in train_indexs]
            dev_data   = [data[i] for i in dev_indexs]
            print("filter_count: ", filter_count)
            print("train: ", len(train_data))
            print("dev: ", len(dev_data))
            dataset_dict.update({
                "train": KBQADataset(train_data),
                "dev": KBQADataset(dev_data),
            })
        else:
            print("filter_count: ", filter_count)
            print("test: ", len(data))
            dataset_dict.update({
                "test": KBQADataset(data),
            })
    
    if head != -1:
        for key in dataset_dict.keys():
            dataset_dict[key].data = dataset_dict[key].data[:head]
    
    return dataset_dict, rel_dict
