import torch
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

def test_generation():
    # 初始化加速器
    accelerator = Accelerator()
    
    # 加载模型和分词器
    model_path = "/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    
    # 准备测试输入
    test_input = "请解释这道数学题：小明有5个苹果，吃掉2个，还剩几个？"
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
    
    # 构造batch
    batch = {
        'generate_prefix_kwargs': {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    }
    
    # 测试生成
    try:
        print("开始生成...")
        output = model.generate(
            **batch['generate_prefix_kwargs'],
            max_new_tokens=512,
            num_beams=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 解码输出
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\n生成的文本:")
        print(generated_text)
        
        # 打印输出信息
        print("\n输出形状:", output.shape)
        print("输出类型:", type(output))
        
    except Exception as e:
        print(f"生成时发生错误: {e}")
        print(f"错误类型: {type(e)}")

if __name__ == "__main__":
    test_generation()