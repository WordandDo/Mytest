#!/usr/bin/env python3
"""简单测试 tool_stats 优化"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server.core.tool_stats import ToolStatsCollector
from pathlib import Path
import shutil

# 创建测试目录
test_dir = Path("test_stats_simple")
if test_dir.exists():
    shutil.rmtree(test_dir)

print("创建 ToolStatsCollector...")
collector = ToolStatsCollector(
    output_dir=str(test_dir),
    enable_realtime_save=True,
    save_interval=3
)

print("记录10次工具调用...")
for i in range(10):
    collector.record_call(
        tool_name=f"tool_{i % 3}",
        task_id="test_task",
        success=(i % 5 != 0),
        error_message="Test error" if i % 5 == 0 else None,
        duration_ms=float(i * 10)
    )
    print(f"  调用 {i+1}/10")

print("\n导出报告...")
report_path = collector.export_report()
print(f"报告路径: {report_path}")

print("\n检查文件...")
realtime_log = test_dir / "realtime_calls.jsonl"
if realtime_log.exists():
    with open(realtime_log) as f:
        lines = f.readlines()
    print(f"✅ 实时日志包含 {len(lines)} 条记录")
else:
    print("❌ 实时日志文件不存在")

print("\n清理测试目录...")
shutil.rmtree(test_dir)
print("✅ 测试完成！")
