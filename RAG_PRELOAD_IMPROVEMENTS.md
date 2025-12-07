# RAG 资源预加载改进说明

## 问题背景

原来的 `start_backend.sh` 脚本存在**懒加载问题**：
- 脚本启动后立即返回，但 RAG 索引可能还在后台加载
- 导致后续的测评脚本连接时资源尚未就绪
- 第一次查询会触发索引加载，造成长时间等待或超时

## 改进方案

### 1. 增强的 `start_backend.sh`

**新增功能：**

- **智能健康检查**：监控 RAG 服务的 `/health` 端点，直到 `ready: true`
- **索引加载等待**：最长等待 600 秒（10分钟），确保大型索引完全加载
- **资源预热测试**：自动执行测试查询，验证索引真正可用
- **进度显示**：每隔固定时间显示加载进度，避免用户焦虑
- **彩色输出**：使用颜色区分不同类型的消息（INFO/SUCCESS/WARNING/ERROR）

**使用方法：**

```bash
# 基本用法（前台运行，显示所有日志）
./start_backend.sh

# 后台运行
nohup ./start_backend.sh > logs/backend_startup.log 2>&1 &
```

**输出示例：**

```
[INFO] 🚀 Starting Resource API on port 8000...
[INFO] Waiting for Resource API to start...
[SUCCESS] Resource API is listening on port 8000
[INFO] Waiting for RAG service to be fully ready (index loading)...
[WARNING] RAG service started but index is still loading...
[INFO] Still waiting for RAG index to load... (40s elapsed)
[SUCCESS] RAG service is fully ready (index loaded)
[INFO] Performing resource warmup test...
✅ RAG warmup query successful
[SUCCESS] Resource warmup completed successfully

[SUCCESS] ==========================================
[SUCCESS] Backend Services Ready
[SUCCESS] ==========================================
[INFO] Resource API:  http://localhost:8000
[INFO] RAG Service:   http://localhost:8001
[INFO] Resource API PID: 12345
```

### 2. 独立的资源预热脚本 `warmup_resources.py`

**功能：**

- 全面的健康检查和功能测试
- 支持测试多种检索模式（dense/sparse）
- 详细的性能统计（响应时间、结果数量）
- 灵活的配置选项

**使用方法：**

```bash
# 基本用法
python warmup_resources.py

# 指定超时时间
python warmup_resources.py --timeout 300

# 同时测试稀疏检索
python warmup_resources.py --test-sparse

# 自定义服务 URL
python warmup_resources.py \
  --resource-api-url http://localhost:8000 \
  --rag-service-url http://localhost:8001
```

**输出示例：**

```
============================================================
Resource Warmup Test Suite
============================================================

[INFO] Checking Resource API availability...
[SUCCESS] Resource API is available
[INFO] Resource pools: ['rag_hybrid']
[INFO] Waiting for RAG service to be ready (timeout=600s)...
[INFO] RAG service status: ok, ready: False
[INFO] RAG service status: ok, ready: True
[SUCCESS] RAG service is ready (took 45.3s)

[INFO] Testing RAG query (search_type=dense, top_k=5)...
[INFO] Query: 'What is artificial intelligence?'
[SUCCESS] Query successful (took 1.23s)
[INFO] Retrieved 5 results
[INFO] Top result preview: {'text': 'Artificial intelligence (AI) is intelligence...'}

============================================================
[SUCCESS] ✅ All warmup tests passed!
[SUCCESS] Backend services are fully ready for use.
============================================================
```

## 技术细节

### 3. 关键改进点

#### 3.1 健康检查机制

**位置**：[start_backend.sh:62-94](start_backend.sh#L62-L94)

```bash
# 检查 RAG 服务端口
if nc -z localhost $RAG_SERVICE_PORT 2>/dev/null; then
    # 检查健康状态
    health_response=$(curl -s http://localhost:${RAG_SERVICE_PORT}/health 2>/dev/null)

    # 提取 ready 字段
    ready_status=$(echo "$health_response" | grep -o '"ready":\s*true')

    if [ -n "$ready_status" ]; then
        # 索引已完全加载
        break
    fi
fi
```

**原理**：
- RAG 服务启动时 `ready: false`
- 索引加载完成后 `ready: true`
- 通过轮询检测 `ready` 状态变化

#### 3.2 资源预热查询

**位置**：[start_backend.sh:101-124](start_backend.sh#L101-L124)

```bash
# 执行实际查询，触发所有延迟初始化
python -c "
import requests
response = requests.post(
    'http://localhost:${RAG_SERVICE_PORT}/query',
    json={'query': 'test warmup query', 'top_k': 1, 'search_type': 'dense'},
    timeout=30
)
if response.status_code == 200:
    print('✅ RAG warmup query successful')
"
```

**作用**：
- 触发任何剩余的延迟初始化
- 验证查询管道完整可用
- 预热缓存和模型

#### 3.3 配置优化建议

**调整 `deployment_config.json` 中的超时时间：**

```json
{
  "resources": {
    "rag_hybrid": {
      "enabled": true,
      "config": {
        "server_start_retries": 600,  // 从 30 增加到 600（10分钟）
        ...
      }
    }
  }
}
```

**原因**：
- 大型 RAG 索引（如 E5 + BM25）加载需要更多时间
- GPU 索引初始化需要分配显存
- 避免因超时导致的假失败

## 使用场景

### 场景 1：开发调试

```bash
# 终端 1：启动后端（前台，看详细日志）
./start_backend.sh

# 终端 2：等待启动完成后，运行测评
./run_rag_env_multimode.sh hybrid
```

### 场景 2：生产环境

```bash
# 后台启动，日志重定向
nohup ./start_backend.sh > logs/backend.log 2>&1 &

# 等待几分钟，检查就绪状态
python warmup_resources.py

# 确认就绪后，启动测评
./run_rag_env_multimode.sh all
```

### 场景 3：CI/CD 管道

```bash
#!/bin/bash
set -e  # 遇到错误立即退出

# 启动后端服务
./start_backend.sh &
BACKEND_PID=$!

# 等待就绪
python warmup_resources.py || {
    echo "Backend warmup failed"
    kill $BACKEND_PID
    exit 1
}

# 运行测评
./run_rag_env_multimode.sh all

# 清理
kill $BACKEND_PID
```

## 配置参数说明

### start_backend.sh 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RESOURCE_PORT` | 8000 | Resource API 端口 |
| `RAG_SERVICE_PORT` | 8001 | RAG 服务端口 |
| `MAX_WAIT_TIME` | 600 | 最大等待时间（秒）|
| `HEALTH_CHECK_INTERVAL` | 2 | 健康检查间隔（秒）|

### warmup_resources.py 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timeout` | 600 | 最大等待时间（秒）|
| `--test-sparse` | False | 是否测试稀疏检索 |
| `--resource-api-url` | http://localhost:8000 | Resource API URL |
| `--rag-service-url` | http://localhost:8001 | RAG Service URL |

## 故障排查

### 问题 1：RAG 服务一直不就绪

**症状**：
```
[WARNING] RAG service did not become ready within 600s
```

**可能原因**：
1. 索引文件路径错误（检查 `deployment_config.json`）
2. GPU 显存不足（检查 `nvidia-smi`）
3. 索引文件损坏（尝试重建索引）

**解决方法**：
```bash
# 检查 RAG 服务日志
curl http://localhost:8001/health

# 查看详细错误
tail -f logs/resource_api.log
```

### 问题 2：预热查询失败

**症状**：
```
❌ RAG warmup query failed: HTTPError 500
```

**可能原因**：
1. 索引格式与配置不匹配
2. 模型文件缺失或损坏
3. 依赖库版本不兼容

**解决方法**：
```bash
# 手动测试查询
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 1, "search_type": "dense"}'

# 查看详细错误堆栈
python warmup_resources.py --timeout 30
```

### 问题 3：端口被占用

**症状**：
```
[ERROR] Address already in use: 8001
```

**解决方法**：
```bash
# 查找占用端口的进程
lsof -i :8001

# 杀死进程
kill -9 <PID>

# 或使用脚本自动清理
./start_backend.sh  # 脚本会自动清理
```

## 性能优化建议

### 1. 索引预加载

在 `deployment_config.json` 中启用：

```json
{
  "config": {
    "preload_index": true,  // 启动时立即加载索引
    "use_gpu_index": true,  // 使用 GPU 加速
    "gpu_parallel_degree": 2  // GPU 并行度
  }
}
```

### 2. 调整超时时间

根据索引大小调整：

| 索引大小 | 推荐超时 |
|----------|----------|
| < 1GB | 60s |
| 1-5GB | 300s |
| 5-10GB | 600s |
| > 10GB | 900s |

### 3. 启用缓存

在查询参数中启用结果缓存：

```python
# 在 rag_pool.py 中添加缓存层
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query: str, top_k: int, search_type: str):
    return rag_index_instance.query(query, top_k, search_type)
```

## 总结

通过这些改进，`start_backend.sh` 现在能够：

✅ 确保 RAG 索引完全加载后才返回
✅ 提供清晰的加载进度反馈
✅ 自动执行资源预热测试
✅ 避免测评脚本遇到未就绪的服务
✅ 支持灵活的配置和故障排查

**关键改进**：从"启动即返回"变为"就绪后返回"，彻底解决懒加载问题。
