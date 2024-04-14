# File Structure

├── data (Unzip the data file)

├── log (Initially an empty folder)

├── models (Initially an empty folder)

├── src

├── tool

├── run.sh

# Dataset

url: https://pan.baidu.com/s/1DyHfVYwecN5IIHWYzxFr7Q?pwd=lyjl 

extraction code: lyjl 

or

url: https://drive.google.com/file/d/1bxQvThcPfnMpblYGzfHsI5BPIFxBDQH_/view?usp=drive_link

# Usage

```bash
bash run.sh [name] [description] [device]
```


## Math23K

```bash
bash run.sh train_math23k TrainMath23K 0
```

```bash
bash run.sh train_math23k_5fold TrainMath23K5fold 0
```

## MathQA

```bash
bash run.sh train_math23k TrainMathQA 0
```

## MAWPS

```bash
bash run.sh train_mawps TrainMAWPS 0
```

## CMWPA

```bash
bash run.sh train_template TrainCMWPA 0
```

```bash
bash run.sh train_template_binary TrainCMWPABinary 0
```
