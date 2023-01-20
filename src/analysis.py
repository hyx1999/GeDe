from dataset import loadMathQA, loadMath23K, loadMAWPS, loadTemplate

train_dataset, test_dataset, const_nums = loadMAWPS('../data/MAWPS', 0)
cnt = len(train_dataset) + len(test_dataset)
op_cnt = 0
dl_cnt = 0
for x in train_dataset + test_dataset:
    op_cnt += len(x["Expr_list"])
    dl_cnt += len(" ".join(x["seg_text"]))

print("MAWPS")
print("avg. #operation", op_cnt / cnt)
print("avg. #PDL", dl_cnt / cnt)

train_dataset, dev_dataset, test_dataset, const_nums = loadMathQA('../data/MathQA')
cnt = len(train_dataset) + len(dev_dataset) + len(test_dataset)
op_cnt = 0
dl_cnt = 0
for x in train_dataset + dev_dataset + test_dataset:
    op_cnt += len(x["Expr_list"])
    dl_cnt += len(" ".join(x["seg_text"]))

print("MathQA")
print("avg. #operation", op_cnt / cnt)
print("avg. #PDL", dl_cnt / cnt)

train_dataset, dev_dataset, test_dataset, const_nums = loadMath23K('../data/Math23K')
cnt = len(train_dataset) + len(dev_dataset) + len(test_dataset)
op_cnt = 0
dl_cnt = 0
for x in train_dataset + dev_dataset + test_dataset:
    op_cnt += len(x["Expr_list"])
    dl_cnt += len(" ".join(x["seg_text"]))

print("Math23K")
print("avg. #operation", op_cnt / cnt)
print("avg. #PDL", dl_cnt / cnt)


train_dataset, dev_dataset, test_dataset = loadTemplate('../data/MathTemplate')
cnt = len(train_dataset) + len(dev_dataset) + len(test_dataset)
op_cnt = 0
dl_cnt = 0
for x in train_dataset + dev_dataset + test_dataset:
    op_cnt += len(x["Expr_list"])
    dl_cnt += len(" ".join(x["seg_text"]))
print("CMWPA")
print("avg. #operation", op_cnt / cnt)
print("avg. #PDL", dl_cnt / cnt)
