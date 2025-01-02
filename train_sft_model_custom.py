import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import Trainer, DataCollatorForSeq2Seq
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

# 初始化分布式环境
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # 使用环境变量初始化
            world_size=world_size,
            rank=rank
        )
        return local_rank
    return 0

# 设置分布式环境
local_rank = setup_distributed()

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 初始化 Accelerator（在进程组初始化之后）
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=8,
    mixed_precision="bf16",
    kwargs_handlers=[ddp_kwargs]
)

# 配置训练参数（确保在进程组初始化之后）
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    save_steps=100,
    logging_steps=10,
    gradient_checkpointing=True,
    # 分布式训练相关参数
    local_rank=local_rank,
    ddp_backend="nccl",
    dataloader_num_workers=4
)

# 设置模型和数据路径
model_path = "/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat"
data_path = "/home/wangxinrong/workspace/reft/divination/mwp_ReFT/data/my_data/zhouyi_train.json"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动处理设备映射
)

# 加载数据集
dataset = load_dataset('json', data_files=data_path)

# 数据预处理函数
def preprocess_function(examples):
    # 根据实际数据格式调整
    inputs = examples["question"]
    targets = examples["answer_cot"]
    
    model_inputs = tokenizer(inputs, max_length=2048, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=2048, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
processed_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset["train"].column_names,
    num_proc=4,
)

# 配置优化器、数据加载器和学习率调度器
optimizer = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

train_dataloader = DataLoader(
    processed_dataset["train"], 
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True,
    num_workers=training_args.dataloader_num_workers,
    pin_memory=True
)

# 计算总训练步数
num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
num_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs

# 配置学习率调度器
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 10,  # 10% 的热身步骤
    num_training_steps=num_training_steps
)

# 使用 accelerator 准备所有组件
model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

# 使用 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer),
)

# 使用 accelerator 准备 trainer
trainer = accelerator.prepare(trainer)

# 开始训练
trainer.train()

# 在程序结束时清理
if dist.is_initialized():
    dist.destroy_process_group()

