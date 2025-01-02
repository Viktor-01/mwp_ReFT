cd /home/wangxinrong/workspace/reft/divination/mwp_ReFT
conda activate cuda-11.7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
--config_file /home/wangxinrong/.cache/huggingface/accelerate/default_config.yaml \
    train_test.py