import numpy as np
import os
import json
from torch.utils.data import Dataset
from typing import Dict, AnyStr, List, Union, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict
from kbqa_utils import Expr, KBQADataset, RawDataInstance


def loadWebQSP(path: AnyStr, head: Optional[int] = None) -> Dict[AnyStr, KBQADataset]:
    dataset_dict: Dict[str, KBQADataset] = dict()
    rel_dict  = defaultdict(set)
    type_dict = defaultdict(set)
    max_edge_num = 0
    for key in ["train", "test"]:
        data_path = os.path.join(path, "data", "webqsp_0107.{}.json".format(key))
        data = []
        filter_count = 0
        with open(data_path, "r") as f:
            obj = json.loads(f.read())
            for raw_item in obj:
                qid = raw_item["qid"]
                query = raw_item["question"]
                S_expr = raw_item["s_expression"]
                answer = [x["answer_argument"] for x in raw_item["answer"]]
                if S_expr is None:
                    filter_count += 1
                    continue
                relations = []
                types = []
                for edge in raw_item["graph_query"]["edges"]:
                    relations.append(edge["relation"])
                for node in raw_item["graph_query"]["nodes"]:
                    types.append(node["class"])
                relations = list(set(relations))
                item = RawDataInstance(qid, query, S_expr, relations, answer)
                data.append(item)
                for rel in relations:
                    rel_dict[key].add(rel)
                for cls in types:
                    type_dict[key].add(cls)
                max_edge_num = max(max_edge_num, len(raw_item["graph_query"]["edges"]))

        if key == "train":
            dev_indexs = np.random.choice(len(data), 100, replace=False).tolist()
            train_indexs = [i for i in range(len(data)) if i not in dev_indexs]

            train_data = [data[i] for i in train_indexs]
            dev_data   = [data[i] for i in dev_indexs]
            print("filter_count: ", filter_count)
            print("train: ", len(train_data))
            print("dev: ", len(dev_data))
            dataset_dict.update({
                "train": train_data,
                "dev": dev_data,
            })
        else:
            print("filter_count: ", filter_count)
            print("test: ", len(data))
            dataset_dict.update({
                "test": data,
            })

    print("max_edge_num: ", max_edge_num)
    
    if head is not None and head != -1:
        for key in dataset_dict.keys():
            dataset_dict[key] = dataset_dict[key][:head]
    
    return dataset_dict, rel_dict, type_dict
