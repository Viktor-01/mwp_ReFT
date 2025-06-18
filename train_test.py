import os
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime,  timedelta
import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass   

# 自定义数据集类
class ZhouYiDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 分别构建输入和输出部分
        instruction = f"问题：{item['question']}\n\n请根据周易卦象分析并给出回答。\n\n"
        response = f"回答：{item['answer_cot']}"
        
        # 分别编码
        input_ids = self.tokenizer(
            instruction,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        labels = self.tokenizer(
            response,
            add_special_tokens=False,  # 不添加特殊token
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # 构建完整的输入序列
        full_input_ids = torch.cat([input_ids, labels])
        
        # 确保不超过最大长度
        if len(full_input_ids) > self.max_length:
            full_input_ids = full_input_ids[:self.max_length]
        
        # 创建attention mask
        attention_mask = torch.ones_like(full_input_ids)
        
        # 创建labels（将非回答部分设置为-100）
        full_labels = torch.full_like(full_input_ids, -100)
        full_labels[len(input_ids):] = labels[:self.max_length-len(input_ids)]
        
        # padding处理
        padding_length = self.max_length - len(full_input_ids)
        if padding_length > 0:
            full_input_ids = torch.cat([
                full_input_ids, 
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])
            attention_mask = torch.cat([
                attention_mask, 
                torch.zeros(padding_length)
            ])
            full_labels = torch.cat([
                full_labels, 
                torch.full((padding_length,), -100)
            ])
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": full_labels
        }
        
def do_checkpoint(model, args):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
)

def print_accelerator_info(accelerator):
    print("\n=== Accelerator 配置信息 ===")
    print(f"设备: {accelerator.device}")
    print(f"进程数: {accelerator.num_processes}")
    print(f"是否是主进程: {accelerator.is_main_process}")
    print(f"是否使用分布式训练: {accelerator.distributed_type}")
    print(f"混合精度类型: {accelerator.mixed_precision}")
    
    if hasattr(accelerator.state, 'deepspeed_plugin'):
        print("\n=== DeepSpeed 配置 ===")
        print(accelerator.state.deepspeed_plugin)

def train():
    # 在函数内部初始化 accelerator
    accelerator = Accelerator()
    
    args = TrainingArguments(
        output_dir="/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_test",
    )
    
    if accelerator.is_main_process:
        print_accelerator_info(accelerator)
    
    # 加载模型和分词器
    model_path = "/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载数据集
    dataset = load_dataset(
        "json", 
        data_files="/home/wangxinrong/workspace/reft/divination/mwp_ReFT/data/my_data/zhouyi_train.json"
    )["train"]

    # 创建数据集实例
    train_dataset = ZhouYiDataset(dataset, tokenizer)

    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 学习率调度器
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 准备训练
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

    # 保存模型
    do_checkpoint(model, args)

def main():
    train()

if __name__ == "__main__":
    main()