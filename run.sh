#!/bin/bash

# 设置Python解释器路径
PYTHON_PATH="/home/triplef/anaconda3/envs/py38/bin/python"

# 设置Python脚本路径
PYTHON_SCRIPT="/home/triplef/code/trainableGP/main.py"

# 设置命令行参数
NETWORK_OPERATIONS="standard"
POPULATION=100
GENERATIONS=30
DATA_NAME="cifar10"
DATA_PATH="dataset/raw/cifar10"
CLASSES_NUMBER=10
EPOCHS=20
BATCH_SIZE=16
SAMPLES_PER_CLASS=10
CROSS_VALIDATION=""

# 执行Python命令
$PYTHON_PATH $PYTHON_SCRIPT \
  --network_operations $NETWORK_OPERATIONS \
  --population $POPULATION \
  --generations $GENERATIONS \
  --data_name $DATA_NAME \
  --data_path $DATA_PATH \
  --classes_number $CLASSES_NUMBER \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --samples_per_class $SAMPLES_PER_CLASS \
  $CROSS_VALIDATION
