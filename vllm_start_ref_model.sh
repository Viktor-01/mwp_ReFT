export CUDA_VISIBLE_DEVICES=6,7
eval "$(conda shell.bash hook)"
export CUDA_HOME="/usr/local/cuda-12.2"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
conda activate cuda-12.2

cd /home/wangxinrong/workspace/reft/divination/mwp_ReFT

python -m vllm.entrypoints.openai.api_server \
    --model /share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/epoch_25 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 10240 \
    --port 7999

# # test
# "我想算一下我今年能不能考上研究生。背景：为这个事我都掉了好几斤肉了但是竞争很激烈，我的属相是龙，今年是鸡年。我考的是冶金学"
curl http://localhost:7999/v1/chat/completions   -H "Content-Type: application/json"   -d '{  
    "model": "/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/epoch_25",
    "messages": [  
      {"role": "user", "content": "占事：已婚女占婚姻。"}  
    ],                                                                            
    "temperature": 0.7,   
    "max_tokens": 8192
  }'