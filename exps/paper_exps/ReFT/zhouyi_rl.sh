### GSM8K 
## Python SDP
# Codellama
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exp_name="zhouyi_rl" \
train_file='data/my_data/zhouyi_train.json' \
test_file='data/my_data/zhouyi_test.json' \
engine='nl' \
model_name_or_path='/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/best/' \
ref_model_name_or_path='/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/best/' \
tokenizer_name_or_path='/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/best/' \
n_epochs='1' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template.sh
