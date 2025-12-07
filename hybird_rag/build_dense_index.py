import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
import faiss
import numpy as np
 # 如果需要使用私有模型，可以设置 HF_TOKEN 环境变量
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 配置
model_name = "intfloat/e5-base-v2"
corpus_path = "corpus.jsonl"
output_index_path = "e5_Flat.index"

# 1. 加载模型
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 2. 读取语料并编码
texts = []
print("Reading corpus...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            # E5 要求 passage 前缀（视具体模型而定，e5-base-v2 通常不需要 passage: 前缀用于存库，但在 query 时需要 query: 前缀）
            # 这里简单拼接 title 和 content
            text = f"{obj.get('title', '')}\n{obj.get('contents', '')}".strip()
            texts.append(text)

print(f"Encoding {len(texts)} documents...")
batch_size = 16
all_embeddings = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(embeddings.cpu().numpy())

# 3. 构建 Faiss 索引
final_embeddings = np.vstack(all_embeddings).astype(np.float32)
d = final_embeddings.shape[1]
print(f"Building Faiss index (dim={d})...")

# 使用 Inner Product (IP) 配合归一化向量 = 余弦相似度
index = faiss.IndexFlatIP(d) 
index.add(final_embeddings)

# 4. 保存
print(f"Saving index to {output_index_path}...")
faiss.write_index(index, output_index_path)
print("Done!")