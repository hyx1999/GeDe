import re
import os
import json

file_path = "/data/hyx/projects/mwp/data/GSM8k"

file_path = os.path.join(file_path, "grade_school_math", "data")

test_dataset = []
with open(os.path.join(file_path, "test.jsonl"), "r") as f:
    for line in f.readlines():
        test_dataset.append(json.loads(line))

print(test_dataset[0])

question_text: str = test_dataset[0]["question"]
answer_text: str = test_dataset[0]["answer"]

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
        nums.append(nums_frac[i_frac])
        i_frac += 1
    elif q_texts[i + 1].startswith('[float]'):
        nums.append(nums_float[i_float])
        i_float += 1
    elif q_texts[i + 1].startswith('[int]'):
        nums.append(nums_int[i_int])
        i_int += 1

question_text = "".join(new_q_text)

print(question_text)

p = re.compile('<<[^<>]*>>')
OpSeq_list = re.findall(p, answer_text)
