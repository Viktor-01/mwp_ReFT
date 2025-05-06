import re

def parse_shell_script(script_path):
    # 初始化参数和命令字典
    params = {}
    commands = {}
    
    # 读取shell脚本内容
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    # 解析每一行
    for line in lines:
        line = line.strip()
        
        # 跳过空行和注释行
        if not line or line.startswith('#'):
            continue
            
        # 解析export命令
        if line.startswith('export'):
            match = re.match(r'export\s+(\w+)=(.+)', line)
            if match:
                key, value = match.groups()
                commands[key] = value
            continue
            
        # 解析参数赋值（形如 var=${var:-'default'} 或 var='value'）
        match = re.match(r'(\w+)=\${?([^}]*)}?', line)
        if match:
            key, value = match.groups()
            # 处理带有默认值的情况 ${var:-'default'}
            if ':-' in value:
                value = value.split(':-')[1].strip("'\"")
            params[key] = value
            continue
            
        # 解析直接赋值（形如 var=value）
        match = re.match(r'(\w+)=[\'"]?([^\'"]*)[\'"]?', line)
        if match:
            key, value = match.groups()
            params[key] = value
            
    return params, commands

# 读取两个脚本
template_path = '/home/wangxinrong/workspace/reft/divination/mwp_ReFT/exps/paper_exps/SFT/_template.sh'
zhouyi_path = '/home/wangxinrong/workspace/reft/divination/mwp_ReFT/exps/paper_exps/SFT/zhouyi_sft.sh'

# 先读取模板脚本
template_params, template_commands = parse_shell_script(template_path)

# 再读取zhouyi脚本
zhouyi_params, zhouyi_commands = parse_shell_script(zhouyi_path)

# 合并参数和命令，让zhouyi的值覆盖template的值
final_params = {**template_params, **zhouyi_params}
final_commands = {**template_commands, **zhouyi_commands}

# 打印最终结果
print("最终参数:")
for key, value in final_params.items():
    print(f"{key}: {value}")

print("\n最终命令:")
for key, value in final_commands.items():
    print(f"{key}: {value}")

# 如果需要，你可以将最终的参数传递给main函数
# main(final_params)