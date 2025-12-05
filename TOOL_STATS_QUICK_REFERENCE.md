# 工具调用统计 - 快速参考

## 📊 数据持久化机制

```
┌─────────────────────────────────────────────────────────────────┐
│                      工具调用统计系统                              │
└─────────────────────────────────────────────────────────────────┘

工具被调用
    │
    ├─> 立即记录到内存
    │   ├─ 任务统计
    │   ├─ 工具统计
    │   └─ 详细记录列表
    │
    └─> 每 N 次调用 (默认 10 次)
        │
        └─> 追加到 realtime_calls.jsonl  ✅ 持久化
            │
            └─ JSONL 格式，每行一个 JSON
               {"tool_name": "xxx", "task_id": "xxx", ...}

服务器关闭 (Ctrl+C 或 kill)
    │
    └─> 生成 tool_stats_report_*.json  ✅ 完整报告
        │
        └─ 包含汇总统计 + 所有详细记录
```

## 🔄 数据保留情况

| 场景          | 内存数据 | 实时日志 (JSONL) | 完整报告 (JSON) |
|---------------|----------|------------------|----------------|
| 🟢 正常运行   | ✅ 保留  | ✅ 持续写入      | ❌ 未生成      |
| 💥 进程崩溃   | ❌ 丢失  | ✅ 已写入部分保留 | ❌ 丢失        |
| 🛑 正常关闭   | ✅ 保留  | ✅ 完整保存      | ✅ 生成        |
| 🔄 重启服务器 | ❌ 清空  | ✅ 可加载恢复    | ✅ 保留历史    |

## 📁 文件结构

```
tool_stats/
│
├── realtime_calls.jsonl                    # 当前运行的实时日志
│   └─ 格式: 每行一个 JSON 对象 (JSONL)
│   └─ 写入: 每 10 次调用追加一次
│   └─ 特点: 持久化，进程安全
│
├── realtime_calls_backup_20240115_143000.jsonl  # 旧日志备份
│   └─ 重启时自动备份上次的日志
│
├── tool_stats_report_20240115_143000.json  # 完整报告 1
├── tool_stats_report_20240115_150000.json  # 完整报告 2
└── tool_stats_report_20240115_160000.json  # 完整报告 3
    └─ 格式: 美化的 JSON (带缩进)
    └─ 生成: 每次正常关闭时
    └─ 内容: 汇总统计 + 详细记录
```

## 🎯 关键特性对比

| 特性           | 实时日志 (JSONL)      | 完整报告 (JSON)       |
|----------------|----------------------|----------------------|
| **格式**       | JSONL (每行一个JSON) | 美化 JSON (带缩进)    |
| **写入时机**   | 运行时持续追加        | 关闭时一次性生成      |
| **文件大小**   | 较大 (原始记录)      | 较大 (包含统计+记录)  |
| **易读性**     | 一般 (需工具解析)    | 好 (易于阅读)        |
| **安全性**     | 高 (持久化)          | 中 (需正常关闭)      |
| **用途**       | 数据恢复、实时分析    | 完整报告、归档        |

## 💡 快速命令

### 查看统计

```bash
# 查看摘要
python view_tool_stats.py summary

# 列出所有任务
python view_tool_stats.py list-tasks

# 查看失败的调用
python view_tool_stats.py failures

# 导出报告
python view_tool_stats.py export
```

### 分析实时日志

```bash
# 统计总调用次数
wc -l tool_stats/realtime_calls.jsonl

# 查看失败的调用 (需要 jq)
cat tool_stats/realtime_calls.jsonl | jq 'select(.success == false)'

# 统计每个工具的调用次数
cat tool_stats/realtime_calls.jsonl | jq -r '.tool_name' | sort | uniq -c

# 查看特定任务
cat tool_stats/realtime_calls.jsonl | jq 'select(.task_id == "task_001")'
```

## ⚙️ 配置选项

### 启动服务器

```bash
# 默认配置（推荐）
python src/mcp_server/main.py --config config.json

# 自定义统计目录
python src/mcp_server/main.py --config config.json --stats-dir ./my_stats

# 禁用统计（不推荐）
python src/mcp_server/main.py --config config.json --enable-stats false
```

### 代码配置

```python
from mcp_server.core.tool_stats import ToolStatsCollector

# 默认配置 - 推荐
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=True,  # 启用实时保存
    save_interval=10             # 每 10 次调用保存
)

# 更频繁保存 - 更安全
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=True,
    save_interval=1  # 每次调用都保存
)

# 仅内存 - 性能最好但不安全
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=False
)
```

## 🔍 常见使用场景

### 1. 监控工具健康度

```bash
# 查看工具统计
python view_tool_stats.py tool

# 查看特定工具
python view_tool_stats.py tool --tool-name search_documents
```

### 2. 调试任务失败

```bash
# 查看特定任务
python view_tool_stats.py task --task-id task_001

# 查看该任务的失败调用
python view_tool_stats.py failures --task-id task_001
```

### 3. 性能分析

```python
# 从实时日志分析耗时
import json

total_duration = 0
count = 0

with open('tool_stats/realtime_calls.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line.strip())
        if record['duration_ms']:
            total_duration += record['duration_ms']
            count += 1

avg_duration = total_duration / count if count > 0 else 0
print(f"Average duration: {avg_duration:.2f}ms")
```

### 4. 生成定期报告

```bash
#!/bin/bash
# 定期导出报告脚本

DATE=$(date +%Y%m%d_%H%M%S)
python view_tool_stats.py export --output "report_${DATE}.json"

# 备份实时日志
cp tool_stats/realtime_calls.jsonl "backup_${DATE}.jsonl"
```

## 📚 相关文档

- [TOOL_STATS_USAGE.md](TOOL_STATS_USAGE.md) - 详细使用说明
- [TOOL_STATS_DATA_FORMAT.md](TOOL_STATS_DATA_FORMAT.md) - 数据格式详解
- [tool_stats_example.json](tool_stats_example.json) - 完整报告示例
- [realtime_calls_example.jsonl](realtime_calls_example.jsonl) - 实时日志示例

## ❓ FAQ

**Q: 数据会丢失吗？**
A: 不会。实时日志在运行时持续保存，即使崩溃也不会丢失已记录的数据。

**Q: 会影响性能吗？**
A: 影响很小（< 1%）。默认每 10 次调用才写入一次文件。

**Q: 如何清理旧数据？**
A: 直接删除 `tool_stats/` 目录下的旧文件即可。

**Q: 重启后数据还在吗？**
A: 是的。实时日志会被保留，可以使用 `load_from_realtime_log()` 恢复。
