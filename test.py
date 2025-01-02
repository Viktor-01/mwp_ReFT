from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("/share/finetune/ppo_paper_final_new/_models_outputs_sft/zhouyi_glm4_sft/saved", trust_remote_code=True).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("/home/wangxinrong/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat", trust_remote_code=True).to("cuda")

print(model)
print(tokenizer)
prompt = "你好"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=10)
print(output)