import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datasets import load_from_disk
from torch.utils.data import DataLoader
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_model_path", type=str, required=True,
                        help="Actor模型的路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据的路径")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="output")
    
    # 添加DeepSpeed配置参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def create_ds_config(args):
    """创建DeepSpeed Hybrid Engine配置"""
    return {
        "train_batch_size": args.per_device_train_batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            }
        },
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
        },
        "hybrid_engine": {
            "enabled": True,
            "max_out_tokens": 512,
            "inference_tp_size": 1,
            "release_inference_cache": True,
            "pin_parameters": True,
            "tp_gather_partition_size": 8,
        }
    }

def main():
    args = parse_args()
    
    # 初始化分布式训练
    deepspeed.init_distributed()
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.actor_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.actor_model_path)
    
    # 加载数据集
    dataset = load_from_disk(args.data_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True
    )
    
    # 创建DeepSpeed Hybrid Engine
    ds_config = create_ds_config(args)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    # 训练循环
    engine.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # 前向传播
            outputs = engine(
                input_ids=batch["input_ids"].to(engine.device),
                attention_mask=batch["attention_mask"].to(engine.device),
                labels=batch["labels"].to(engine.device)
            )
            
            loss = outputs.loss
            
            # 反向传播
            engine.backward(loss)
            
            # 优化器步进
            engine.step()
            
            if engine.local_rank == 0 and step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
    
    # 保存模型
    if engine.local_rank == 0:
        engine.save_checkpoint(args.output_dir)

if __name__ == "__main__":
    main()