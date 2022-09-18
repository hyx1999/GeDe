import re
import os
import json

file_path = "/data/hyx/projects/mwp/data/GSM8k"

def load_gsm8k(file_path: str):
    file_path = os.path.join(file_path, "grade_school_math", "data")

    raw_test_dataset = []
    with open(os.path.join(file_path, "test.jsonl"), "r") as f:
        for line in f.readlines():
            raw_test_dataset.append(json.loads(line))

    raw_train_dataset = []
    with open(os.path.join(file_path, "train.jsonl"), "r") as f:
        for line in f.readlines():
            raw_train_dataset.append(json.loads(line))

    const_nums = []

    def parse_data(question_text: str, answer_text: str):

        answer_value = str(eval(answer_text.split("####")[-1]))

        # parse interger
        p0 = re.compile('\d+/\d+')
        p1 = re.compile('\d+\.\d+')
        p2 = re.compile('\d+')

        nums_frac  = re.findall(p0, question_text)
        for num in nums_frac:
            question_text = question_text.replace(num, "[num][frac]")
        nums_float = re.findall(p1, question_text)
        for num in nums_float:
            question_text = question_text.replace(num, "[num][float]")
        nums_int   = re.findall(p2, question_text)
        for num in nums_int:
            question_text = question_text.replace(num, "[num][int]")

        nums = []
        i_frac  = 0
        i_float = 0
        i_int   = 0

        q_texts = question_text.split('[num]')
        new_q_text = [q_texts[0]]
        for i in range(len(q_texts) - 1):
            new_q_text.append('[num{}]'.format(len(num)))
            new_q_text.append(q_texts[i + 1])
            if q_texts[i + 1].startswith('[frac]'):
                nums.append(str(eval(nums_frac[i_frac])))
                i_frac += 1
            elif q_texts[i + 1].startswith('[float]'):
                nums.append(str(eval(nums_float[i_float])))
                i_float += 1
            elif q_texts[i + 1].startswith('[int]'):
                nums.append(str(eval(nums_int[i_int])))
                i_int += 1

        question = "".join(new_q_text)

        p3 = re.compile('<<[^<>]*>>')
        p4 = re.compile('<<([^=<>]*)=([^=<>]*)>>')
        raw_OpSeq_list = re.findall(p3, answer_text)
        
        all_nums = [x for x in nums]
        OpSeq_list = []
        for opseq_text in raw_OpSeq_list:
            m = p4.match(opseq_text)
            if m is None:
                raise ValueError
            v0, v1 = m.group(1, 2)
            raw_expr_toks = re.split(r"([\*\/\+\-])", v0)
            expr_toks = []
            for x in raw_expr_toks:
                if x in "+-*/":
                    expr_toks.append(x)
                else:
                    x = str(eval(x))
                    if x in all_nums:
                        expr_toks.append('[num{}]'.format(all_nums.index(x)))
                    else:
                        if x not in const_nums:
                            const_nums.append(x)
                        expr_toks.append('[c{}]'.format(const_nums.index(x)))
            all_nums.append(str(eval(v1)))
            OpSeq_list.append({
                "arg0": len(all_nums) - 1,
                "expr_toks": expr_toks,
                "expr_str": "".join(expr_toks)
            })

        if answer_value != all_nums[-1]:
            return None
        else:
            return {
                "question": question,
                "nums": nums,
                "OpSeq_list": OpSeq_list,
            }

    train_dataset = []
    test_dataset  = []

    for dataset, raw_dataset in zip([train_dataset, test_dataset], [raw_train_dataset, raw_test_dataset]):
        for raw_obj in raw_dataset:
            obj = parse_data(raw_obj["question"], raw_obj["answer"])
            if obj is not None:
                obj["const_nums"] = const_nums
                dataset.append(obj)
    
    return train_dataset, test_dataset
