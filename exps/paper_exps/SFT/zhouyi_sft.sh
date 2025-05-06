

## NL
# Codellama
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
conda init bash
source ~/.bashrc
conda activate cuda-11.7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
exp_name="zhouyi_glm4_sft" \
train_file='data/my_data/zhouyi_train.json' \
test_file='data/my_data/zhouyi_test.json' \
engine='nl' \
model_name_or_path='/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat' \
tokenizer_name_or_path='/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat' \
n_epochs='40' \
    bash exps/paper_exps/SFT/_template.sh
