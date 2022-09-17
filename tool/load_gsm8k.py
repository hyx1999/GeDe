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

question_text = test_dataset[0]["question"]
answer_text = test_dataset[0]["answer"]

# parse interger
p0 = re.compile('\d+/\d+')
p1 = re.compile('\d+\.\d+')
p2 = re.compile('\d+')

nums_frac  = re.findall(p0)
nums_float = re.findall(p1)
nums_int   = re.findall(p2)



p = re.compile('<<[^<>]*>>')
print(re.findall(p1, answer_text))
