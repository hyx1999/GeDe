from dataset import loadMathQA, loadMath23K

train_dataset, dev_dataset, test_dataset, const_nums = loadMathQA('../data/MathQA')
count = [0 for _ in range(20)]
for x in test_dataset:
    count[len(x["Expr_list"])] += 1

print("MathQA")
print(count, sum(count))


train_dataset, dev_dataset, test_dataset, const_nums = loadMath23K('../data/Math23K')
count = [0 for _ in range(20)]
for x in test_dataset:
    count[len(x["Expr_list"])] += 1

print("Math23K")
print(count, sum(count))
