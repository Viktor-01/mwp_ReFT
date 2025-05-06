import requests
import json

# 获取特定版本的依赖信息
version = "0.6.4"
response = requests.get(f"https://pypi.org/pypi/vllm/{version}/json")
data = response.json()

# 打印所有依赖
print(f"Dependencies for vllm {version}:")
for req in data["info"]["requires_dist"]:
    print(req)