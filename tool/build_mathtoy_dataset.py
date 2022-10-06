from tqdm import tqdm
import random
import os
import json

random.seed(0)

if not os.path.exists("../data/MathToy"):
    os.makedirs("../data/MathToy")

def genDAGInstance(id: str):
    base_num_cnt = random.randint(3, 5)
    num_cnt = random.randint(base_num_cnt + 1, 10)
    base_nums = [str(random.randint(1, 100)) for _ in range(base_num_cnt)]

    texts = ["已知 {} .".format(
        " ".join([f"[v{i}] 等于 [num{i}]," for i, _ in enumerate(base_nums)])
    )]
    
    OpSeq_list = []
    expr_toks_list = [[f'[num{i}]'] for i in range(base_num_cnt)]
    for i in range(base_num_cnt, num_cnt):
        mode = random.randint(1, 3)
        if mode == 1:
            j = i - 1
            texts.append(f"[v{i}] 等于 [v{j}].")
            OpSeq_list.append({
                "arg0": i,
                "expr_toks": [f'[num{j}]']
            })
            expr_toks_list.append(expr_toks_list[j])
        elif mode == 2:
            js = [random.randint(0, i - 1), i - 1]
            op = random.choice(['+', '-'])
            if op == '+':
                texts.append(f"[v{i}] 等于 [v{js[0]}] 与 [v{js[1]}] 的和.")
            else:
                texts.append(f"[v{i}] 等于 [v{js[0]}] 与 [v{js[1]}] 的差.")
            OpSeq_list.append({
                "arg0": i,
                "expr_toks": [f'[num{js[0]}]', op, f'[num{js[1]}]']
            })
            expr_toks_list.append(['('] + expr_toks_list[js[0]] + [op] + expr_toks_list[js[1]] + [')'])
        else:
            js = [random.randint(0, i - 1), random.randint(0, i - 1), i - 1]
            op0 = random.choice(['+', '-'])
            if op0 == '+':
                text = f"[v{i}] 等于 [v{js[0]}] 与 [v{js[1]}] 的和"
            else:
                text = f"[v{i}] 等于 [v{js[0]}] 与 [v{js[1]}] 的差"
            op1 = random.choice(['+', '-'])
            if op1 == '+':
                text = text + f'再加 [v{js[2]}] .'
            else:
                text = text + f'再减 [v{js[2]}] .'
            OpSeq_list.append({
                "arg0": i,
                "expr_toks": [f'[num{js[0]}]', op0, f'[num{js[1]}]', op1, f'[num{js[2]}]']
            })
            expr_toks_list.append(
                ['('] + expr_toks_list[js[0]] + [op0] + expr_toks_list[js[1]] + [op1] + expr_toks_list[js[2]] + [')'])
            texts.append(text)
    texts.append(f"求[v{num_cnt - 1}].")
    return {
        "sample_id": id,
        "raw_text": "".join(texts),
        "seg_text": "".join(texts).split(' '),
        "nums": base_nums,
        "const_nums": [],
        "OpSeq_list0": OpSeq_list,
        "OpSeq_list1": {
            "arg0": base_num_cnt,
            "expr_toks": expr_toks_list[-1]
        }
    }

folder_path = "../data/MathToy"

train_path = os.path.join(folder_path, "train.json")
dev_path = os.path.join(folder_path, "dev.json")
test_path = os.path.join(folder_path, "test.json")

for path, num in zip([train_path, dev_path, test_path], [10000, 1000, 1000]):
    data = [genDAGInstance(str(_)) for _ in tqdm(range(num), total=num)]
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False))
