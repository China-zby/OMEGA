#!/bin/bash

# 设置运行的次数和每次运行间隔
repeats=10
interval=3

# 循环运行 Python 脚本
for ((i=9; i<=repeats; i++))
do
    echo "Running iteration $i"
    python mix_reinforce_dpn_new.py --input_epoch $i --query_path '/home/lzp/zby/query_datasets/query_total_car' --video_path '/home/lzp/zby/youtubu-total.mp4'  --cache_path /home/lzp/zby/opencv/cache/youtubu_cache_shell.joblib
done