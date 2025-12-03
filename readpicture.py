import json
import base64
import os

# 读取轨迹文件
with open('results/test_run_hybrid/trajectory.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        task_id = data.get('task_id', 'unknown')
        messages = data.get('messages', [])
        
        save_dir = f"screenshots/{task_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        action_img_idx = 1
        
        for msg_idx, msg in enumerate(messages):
            content_list = msg.get('content')
            role = msg.get('role')
            
            if isinstance(content_list, list):
                # 检查当前消息中文本内容，判断是否是初始观察
                # HttpMCPEnv 中初始观察会伴随文本: "Here is the initial screen state of the computer:"
                is_initial = False
                for item in content_list:
                    if item.get('type') == 'text' and "initial screen state" in item.get('text', ''):
                        is_initial = True
                        break
                
                # 如果没有特定文本，也可以简单判断：如果是第一条包含图片的用户消息，通常就是初始观察
                # if msg_idx == 1 and role == 'user': ... (根据具体 message 结构调整索引)

                for content in content_list:
                    if content.get('type') == 'image_url':
                        url = content['image_url']['url']
                        b64_data = url.split(',')[1]
                        
                        if is_initial:
                            filename = "initial_state.png"
                            print(f"[{task_id}] Found Initial Observation -> saved as {filename}")
                        else:
                            filename = f"action_step_{action_img_idx}.png"
                            print(f"[{task_id}] Found Action Screenshot -> saved as {filename}")
                            action_img_idx += 1
                        
                        with open(os.path.join(save_dir, filename), "wb") as img_file:
                            img_file.write(base64.b64decode(b64_data))