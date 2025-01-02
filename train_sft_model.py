# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass  

import traceback
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import datetime,  timedelta
import time
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code
from src.utils import set_seed, floatify, compute_ETA
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW, get_constant_schedule_with_warmup
import wandb
import pandas as pd
import shutil
import signal
from contextlib import contextmanager
tqdm = partial(tqdm, ncols=0, leave=False)


TIMEOUT = 10
instruction=None
cot_trigger=None
answer_trigger=None
def setup_cot(src_name):
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric', 'zhouyi']
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\n因此，答案是：'
    return 

post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    'mathqa-numeric': lambda x: float(x),
    'zhouyi': lambda x: x.strip(),
}
### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    ('nl', 'mathqa-numeric'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'zhouyi'): lambda answer_cot: [res.split(answer_trigger)[-1].strip() for res in answer_cot],
}
compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'zhouyi': lambda extracted_ans, target_answer: extracted_ans == target_answer,
}

def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'],'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'],'r'))),
        })
        accelerator.print('Raw data:', raw_dataset)
        src_name = raw_dataset['train'][0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        setup_cot(src_name)
        accelerator.print('Using instruction:', instruction)
        accelerator.print('Using cot_trigger:', cot_trigger)
        accelerator.print('Using answer_trigger:', answer_trigger)
        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \

                question = question.strip()
                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot is not None:
                    answer_cot = answer_cot.strip()
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'

                input = f'{instruction}{question}{cot_trigger}'
                output = f'{answer_cot}'
                prefix_text = f'{instruction}{question}{cot_trigger}'

                input_encode = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1]* len(input_ids)
                prefix = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']

                # Truncation
                input_ids_max_length = len(input_ids)
                # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
                input_ids = input_ids[:args['max_input_length']]
                labels = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

                ##
                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                ##
                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)
                new_batch['input_ids_max_length'].append(input_ids_max_length)
            
            return new_batch

        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True, remove_columns=dataset.column_names, 
                num_proc=8, load_from_cache_file=False
            ) for mode, dataset in raw_dataset.items()})
        accelerator.print('Processed data:', tokenized_dataset)
        for mode, dataset in tokenized_dataset.items():
            accelerator.print(mode, f'{mode}_input_ids_max_length', max(dataset['input_ids_max_length']))

        if accelerator.is_main_process and args['wandb_log']:
            wandb.config.update({
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
                "raw_dataset": str(raw_dataset),
                "tokenized_dataset": str(tokenized_dataset),
                "train_input_ids_max_length": max(tokenized_dataset['train']['input_ids_max_length']),
                "test_input_ids_max_length": max(tokenized_dataset['test']['input_ids_max_length']),
            })

    def collate_fn(batch, args, tokenizer):
        max_input_length = max([len(item['input_ids']) for item in batch])
        max_target_length = max([len(item['labels']) for item in batch])
        max_prefix_length = max([len(item['prefix']) for item in batch])
        input_ids  = []
        attention_mask  = []
        labels, labels_left_padded  = [], []
        prefix_left_padded  = []
        prefix_attention_mask_left_padded  = []
        for item in batch:
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
            attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
            labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))

            labels_left_padded.append([-100]*(max_target_length - len(item['labels'])) + item['labels'])
            prefix_left_padded.append([tokenizer.pad_token_id]*(max_prefix_length - len(item['prefix'])) + item['prefix'])
            prefix_attention_mask_left_padded.append([0]*(max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
        forward_kwargs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.BoolTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }
        generate_prefix_kwargs = {
            'input_ids': torch.LongTensor(prefix_left_padded),
            'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            'labels': torch.LongTensor(labels_left_padded)
        }
        return {
            'forward_kwargs': forward_kwargs,
            'generate_prefix_kwargs': generate_prefix_kwargs,
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)

@contextmanager
def timeout_handler(seconds, description="Operation"):
    def _handle_timeout(signum, frame):
        raise TimeoutError(f"{description} timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def do_checkpoint(args, model, tokenizer, save_path, global_step):
    try:
        # 所有进程同步等待
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
        
        # 等待目录操作完成
        accelerator.wait_for_everyone()
        
        # 添加超时控制
        with timeout_handler(60*30, "Checkpoint saving"):  # 保存模型超过30分钟，则为超时
            model.save_checkpoint(save_path)
        
        # 确保所有进程都保存完成
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            accelerator.print(f"成功保存checkpoint: {save_path}")
            
    except TimeoutError as e:
        accelerator.print(f"保存checkpoint超时: {e}")
        raise
    except Exception as e:
        accelerator.print(f"保存checkpoint错误: {e}")
        raise

def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, test_dataset, test_dataloader, 
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    monitor = DistributedTrainingMonitor(accelerator)
    accelerator.wait_for_everyone()
    
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        for idx, batch in t:
            try:
                # 检查数据加载是否停滞
                if monitor.check_dataloader(idx, batch['forward_kwargs']['input_ids'].size(0)):
                    accelerator.print(f"[进程 {accelerator.process_index}] 检测到数据加载器停滞...")
                    
                    # 如果连续多次停滞，考虑跳过当前epoch
                    if monitor.stuck_count >= monitor.max_stuck_attempts:
                        accelerator.print("数据加载器多次停滞，跳过当前epoch")
                        return {"loss": float('inf')}, global_step
                        
                    # 尝试同步和恢复
                    accelerator.wait_for_everyone()
                    continue
                
                # 正常的训练逻辑
                with accelerator.accumulate(model):
                    output = model(**batch['forward_kwargs'])
                    # Get some metrics
                    loss = output[0]
                    result_dict, extra = {}, None
                    # Update
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if clip_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    # model.zero_grad()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    
                if accelerator.sync_gradients:
                    global_step += 1
                    # Step update metric
                    epoch_result_dict['loss'].append(loss.item()) 
                    for k, v in result_dict.items():
                        epoch_result_dict[k].append(v)

                    # Step evaluating
                    eval_log_dict = {}
                    is_best = False
                    if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                        evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                        eval_log_dict.update(evaluate_result_dict)
                        if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                            is_best = True
                            best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                        if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                            summary_log_dict['Eval.Gen.value_accuracy'] = []
                        summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

                    # Step logging
                    train_log_dict = {}
                    if logging_step_freq is not None and global_step % logging_step_freq == 0:
                        train_log_dict = {f'T.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}
                    
                    if eval_log_dict or train_log_dict:
                        log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                        if accelerator.is_main_process and args['wandb_log']:
                            wandb.log(log_dict, step=global_step)
                            log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                        log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                        accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

                    # Step saving
                    # if saving_step_freq is not None and global_step % saving_step_freq == 0:
                    #     if is_best:
                    #         save_path = os.path.join(model_dir, f'best')
                    #         do_checkpoint(args, model, tokenizer, save_path)
                        # if args['keep_num_ckpt'] > 0:
                        #     save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                        #     do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

                    # Keep only max_record items
                    for k, v in epoch_result_dict.items():
                        if len(v) > 1:
                            epoch_result_dict[k] = v[-1:]
            except Exception as e:
                accelerator.print(f"[进程 {accelerator.process_index}] 批次 {idx} 出错: {e}")
                accelerator.print(traceback.format_exc())
                accelerator.wait_for_everyone()
                continue

    # Metric summary:
    epoch_result_dict = {k:(sum(v)/len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    if accelerator.is_main_process:
        accelerator.print('Evaluating...')
        accelerator.print(f'跳过evaluation，返回0')
        return {'value_accuracy': 0} #TODO: 现在zero3返回的输出都是一样的，而且返回的输出和target格式不一致，需要修改，暂时略去
    else:
        return {'value_accuracy': -1.0}

class DistributedTrainingMonitor:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.last_step = 0
        self.last_update = time.time()
        self.stuck_count = 0
        self.max_stuck_attempts = 3
        
        # 添加数据加载监控相关属性
        self.last_batch_time = time.time()
        self.last_batch_idx = -1
        self.dataloader_stuck_threshold = 300  # 5分钟无数据则认为死锁
        self.batch_times = []  # 记录每个批次的处理时间
        
        
    def check_dataloader(self, batch_idx, batch_size=None):
        """检查数据加载器是否停滞"""
        current_time = time.time()
        time_since_last_batch = current_time - self.last_batch_time
        
        # 检测是否停滞
        if batch_idx == self.last_batch_idx and time_since_last_batch > self.dataloader_stuck_threshold:
            self.accelerator.print(f"[进程 {self.accelerator.process_index}] "
                                f"警告: DataLoader在批次 {batch_idx} 停滞 {time_since_last_batch:.1f} 秒")
            self.stuck_count += 1
            return True
            
        # 更新状态
        if batch_idx != self.last_batch_idx:
            # 记录处理时间
            if batch_size and self.last_batch_idx >= 0:
                batch_time = current_time - self.last_batch_time
                self.batch_times.append(batch_time)
                if len(self.batch_times) > 100:  # 保留最近100个批次的时间
                    self.batch_times.pop(0)
                    
                # 计算平均处理速度
                if self.accelerator.is_main_process and len(self.batch_times) > 0:
                    avg_time = sum(self.batch_times) / len(self.batch_times)
                    samples_per_second = batch_size / avg_time if avg_time > 0 else 0
                    if batch_idx % 10 == 0:  # 每10个批次输出一次
                        self.accelerator.print(f"批次 {batch_idx}: 平均处理速度 {samples_per_second:.1f} samples/s")
            
            self.last_batch_idx = batch_idx
            self.last_batch_time = current_time
            self.stuck_count = 0  # 重置卡住计数
            
        return False
    
    def update(self, step, model, optimizer):
        # 确保所有进程同步
        self.accelerator.wait_for_everyone()
        
        # 使用主进程的时间和状态
        current_time = time.time()
        should_recover = torch.tensor(0, device=self.accelerator.device)
        stuck_count = torch.tensor(self.stuck_count, device=self.accelerator.device)
        
        if self.accelerator.is_main_process:
            if step == self.last_step:
                if current_time - self.last_update > 600:  # 10分钟
                    self.stuck_count += 1
                    should_recover = torch.tensor(1, device=self.accelerator.device)
                    self.accelerator.print(f"警告: 训练在步数 {step} 停滞超过10分钟 (第 {self.stuck_count} 次)")
            else:
                self.last_step = step
                self.last_update = current_time
                self.stuck_count = 0
                stuck_count = torch.tensor(0, device=self.accelerator.device)
        
        # 使用 all_reduce 同步状态
        self.accelerator.wait_for_everyone()
        torch.distributed.all_reduce(should_recover)
        torch.distributed.all_reduce(stuck_count)
        
        # 更新所有进程的计数器
        self.stuck_count = int(stuck_count.item())
        
        if should_recover.item() > 0:
            try:
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 重置优化器状态
                optimizer.zero_grad()
                
                # 如果连续卡住次数过多
                if self.stuck_count >= self.max_stuck_attempts:
                    self.accelerator.print(f"[进程 {self.accelerator.process_index}] 连续多次卡住，采取更激进的恢复措施...")
                    
                    # 同步等待
                    self.accelerator.wait_for_everyone()
                    
                    # 保存临时检查点
                    if self.accelerator.is_main_process:
                        temp_dir = f"temp_recovery_{step}"
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    self.accelerator.wait_for_everyone()
                    
                    # 保存状态
                    self.accelerator.save_state(temp_dir)
                    
                    # 重置梯度
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                    
                    # 加载状态
                    self.accelerator.wait_for_everyone()
                    self.accelerator.load_state(temp_dir)
                    
                    # 清理临时目录
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        shutil.rmtree(temp_dir)
                    
                    # 重置计数器
                    self.stuck_count = 0
                
                return True
                
            except Exception as e:
                self.accelerator.print(f"[进程 {self.accelerator.process_index}] 恢复过程出错: {e}")
                return False
        
        return False

def main(args):
    set_seed(args['seed'] + accelerator.process_index)
    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)
        
    tokenizer = AutoTokenizer.from_pretrained(
        args['tokenizer_name_or_path'],
        trust_remote_code=True,
        padding_side='left',  # ChatGLM 使用左侧填充
        eos_token='<|endoftext|>',  # 设置 EOS token
        pad_token='<|endoftext|>',  # 设置 PAD token
    )
    
    # 确保 tokenizer 有必要的特殊 token
    special_tokens_dict = {
        'pad_token': '<|endoftext|>',
        'eos_token': '<|endoftext|>',
        'bos_token': '<|startoftext|>',
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)
    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        trust_remote_code=True
    )
    
    # 添加缺失的配置
    config._attn_implementation = "eager"  # 添加注意力实现方式
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args['model_name_or_path'],
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # 确保模型参数是 bf16 类型
    model = model.bfloat16()
    
    accelerator.print(f'[Vocab size]: {len(tokenizer)}')    
    model.resize_token_embeddings(len(tokenizer))

    if accelerator.is_main_process and args['wandb_log']:
        wandb.run.summary.update({
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'vocab_size': len(tokenizer)
        })

    n_epochs = args['n_epochs']
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs) // args['gradient_accumulation_steps']
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size']*accelerator.num_processes*args['gradient_accumulation_steps']}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )   
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
    
    global_step = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir=args['model_dir']
    best_eval_log_dict = {}
    summary_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    lowest_loss = None
    monitor = DistributedTrainingMonitor(accelerator)
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=not accelerator.is_main_process) as t:
        for epoch in t:
            kwargs = {
                'args': args,
                'model': model, 
                'train_dataset': train_dataset, 
                'train_dataloader': train_dataloader, 
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer, 
                'scheduler': scheduler,
                'global_step': global_step, 
                'tokenizer': tokenizer,
                'prefix':'', 
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            
            try:
                # 确保所有进程开始新的epoch时是同步的
                accelerator.wait_for_everyone()
                
                with timeout_handler(60*60, f"Epoch {epoch}"): # 训练1轮若超过1小时，则为超时
                    train_epoch_result_dict, global_step = train_one_epoch(**kwargs)
                    accelerator.print(f"[进程 {accelerator.process_index}] Epoch {epoch} 训练结果: {train_epoch_result_dict}")
                    # 检查是否可能死锁
                    if monitor.update(global_step, model, optimizer):
                        accelerator.print(f"[进程 {accelerator.process_index}] Epoch {epoch} 检测到死锁，已尝试恢复")
                        # 同步等待所有进程
                        accelerator.wait_for_everyone()
                        continue  # 继续下一个epoch
                        
            except TimeoutError as e:
                accelerator.print(f"[进程 {accelerator.process_index}] Epoch {epoch} 超时: {e}")
                # 确保所有进程都跳过这个epoch
                accelerator.wait_for_everyone()
                continue
                
            except Exception as e:
                accelerator.print(f"[进程 {accelerator.process_index}] Epoch {epoch} 发生错误: {e}")
                # 严重错误时同步中断所有进程
                raise
                
            # 确保所有进程在epoch结束时同步
            accelerator.wait_for_everyone()
            
            eval_log_dict = {}
            is_best = False
            
            accelerator.print(f'跳过evaluation')
            # if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
            #     evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
            #     eval_log_dict.update(evaluate_result_dict)
            #     if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
            #         is_best = True
            #         best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
            #     if 'Eval.Gen.value_accuracy' not in summary_log_dict:
            #         summary_log_dict['Eval.Gen.value_accuracy'] = []
            #     summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])
            if lowest_loss is None:
                lowest_loss = train_epoch_result_dict["loss"]
                is_best = True
            elif train_epoch_result_dict["loss"] < lowest_loss:
                lowest_loss = train_epoch_result_dict["loss"]
                is_best = True
                
            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {f'T.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in train_epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                accelerator.print(f"[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            # if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
            if is_best:
                save_path = os.path.join(model_dir, f'best')
                if not  accelerator.is_main_process:
                    pass
                else:
                    # 如果目录已存在,先清空
                    if os.path.exists(save_path):
                        accelerator.print(f"目录已存在, 清空目录: {save_path}")
                        shutil.rmtree(save_path)
                    os.makedirs(save_path, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    accelerator.print(f"开始保存新的最佳checkpoint... 时间: {timestamp}")
                    do_checkpoint(args, model, tokenizer, save_path, global_step)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # time.sleep(10)
    # if args['keep_num_ckpt'] > 0:
    #     save_path=os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
    #     do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    return 
if __name__ == '__main__':
    from transformers import HfArgumentParser
    NONE_INT = -100 
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str 
        test_file: str
        batch_size: int = field(default=4)
        eval_batch_size: int = field(default=8)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        max_gen_length: int = field(default=512)
        gradient_accumulation_steps: int = field(default=1)
        keep_num_ckpt: int = field(default=1)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='nl')

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(gradient_accumulation_steps=args['gradient_accumulation_steps'], kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) # wait for processing upto 5hrs
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
