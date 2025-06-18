#!/bin/bash

cd /home/wangxinrong/workspace/reft/divination/mwp_ReFT/

date=$(date +%Y%m%d_%H%M%S)
log_dir="/home/wangxinrong/workspace/reft/divination/mwp_ReFT/logs"
mkdir -p ${log_dir}

# 运行训练脚本
nohup bash /home/wangxinrong/workspace/reft/divination/mwp_ReFT/exps/paper_exps/SFT/zhouyi_sft.sh \
    > ${log_dir}/zhouyi_sft_${date}.log 2>&1 &

# 保存进程ID
echo $! > ${log_dir}/zhouyi_sft_${date}.pid

# 打印启动信息
echo "Training started with PID $!"
echo "Log file: ${log_dir}/zhouyi_sft_${date}.log"