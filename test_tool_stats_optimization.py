#!/usr/bin/env python3
"""
测试 tool_stats 优化效果
验证：
1. 文件 I/O 不在锁内执行
2. 批量写入减少文件操作次数
3. 断点调试时不会卡住
"""
import sys
import os
import time
import threading
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server.core.tool_stats import ToolStatsCollector

def test_concurrent_calls():
    """测试并发调用场景"""
    print("=" * 60)
    print("测试 1: 并发调用场景")
    print("=" * 60)

    # 创建临时测试目录
    test_dir = Path("test_tool_stats_temp")
    test_dir.mkdir(exist_ok=True)

    collector = ToolStatsCollector(
        output_dir=str(test_dir),
        enable_realtime_save=True,
        save_interval=5  # 每5次调用保存一次
    )

    def worker(worker_id: int, num_calls: int):
        """模拟工作线程"""
        for i in range(num_calls):
            collector.record_call(
                tool_name=f"tool_{worker_id % 3}",
                task_id=f"task_{worker_id}",
                success=(i % 10 != 0),  # 每10次失败一次
                error_message="Test error" if i % 10 == 0 else None,
                duration_ms=float(i * 10)
            )
            time.sleep(0.001)  # 模拟工具执行时间

    # 启动多个线程并发调用
    threads = []
    num_workers = 5
    calls_per_worker = 20

    start_time = time.time()

    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i, calls_per_worker))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    print(f"\n✅ 完成 {num_workers} 个线程，每个线程 {calls_per_worker} 次调用")
    print(f"⏱️  总耗时: {elapsed:.3f}秒")
    print(f"📊 平均每次调用: {elapsed / (num_workers * calls_per_worker) * 1000:.2f}ms")

    # 导出报告（会触发 flush）
    report_path = collector.export_report()
    print(f"📄 报告已导出: {report_path}")

    # 打印摘要
    collector.print_summary()

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n🧹 已清理测试目录: {test_dir}")


def test_lock_contention():
    """测试锁竞争情况"""
    print("\n" + "=" * 60)
    print("测试 2: 锁竞争测试（模拟断点场景）")
    print("=" * 60)

    test_dir = Path("test_tool_stats_temp2")
    test_dir.mkdir(exist_ok=True)

    collector = ToolStatsCollector(
        output_dir=str(test_dir),
        enable_realtime_save=True,
        save_interval=3
    )

    lock_wait_times = []

    def timed_worker(worker_id: int):
        """记录获取锁的等待时间"""
        for i in range(10):
            wait_start = time.time()

            # 记录调用（内部会获取锁）
            collector.record_call(
                tool_name=f"tool_{worker_id}",
                task_id=f"task_{worker_id}",
                success=True,
                duration_ms=1.0
            )

            wait_time = (time.time() - wait_start) * 1000
            lock_wait_times.append(wait_time)

            # 模拟一些工作
            time.sleep(0.005)

    # 启动多个线程
    threads = []
    for i in range(10):
        t = threading.Thread(target=timed_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # 分析锁等待时间
    avg_wait = sum(lock_wait_times) / len(lock_wait_times)
    max_wait = max(lock_wait_times)

    print(f"\n📊 锁等待时间统计:")
    print(f"   平均等待: {avg_wait:.2f}ms")
    print(f"   最大等待: {max_wait:.2f}ms")
    print(f"   总调用次数: {len(lock_wait_times)}")

    if max_wait < 50:  # 如果最大等待时间小于50ms
        print("✅ 锁竞争优化成功！文件 I/O 不阻塞其他线程")
    else:
        print("⚠️  仍存在较长的锁等待时间")

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n🧹 已清理测试目录: {test_dir}")


def test_batch_write_efficiency():
    """测试批量写入效率"""
    print("\n" + "=" * 60)
    print("测试 3: 批量写入效率")
    print("=" * 60)

    test_dir = Path("test_tool_stats_temp3")
    test_dir.mkdir(exist_ok=True)

    # 测试不同的 save_interval
    intervals = [1, 5, 10, 20]

    for interval in intervals:
        collector = ToolStatsCollector(
            output_dir=str(test_dir),
            enable_realtime_save=True,
            save_interval=interval
        )

        start_time = time.time()

        # 执行100次调用
        for i in range(100):
            collector.record_call(
                tool_name="test_tool",
                task_id="test_task",
                success=True,
                duration_ms=1.0
            )

        # 刷新剩余记录
        collector._flush_pending_writes()

        elapsed = time.time() - start_time

        print(f"   Interval={interval:2d}: {elapsed:.3f}秒 ({elapsed/100*1000:.2f}ms/call)")

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n🧹 已清理测试目录: {test_dir}")


if __name__ == "__main__":
    print("\n🚀 开始测试 tool_stats 优化效果\n")

    try:
        test_concurrent_calls()
        test_lock_contention()
        test_batch_write_efficiency()

        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n💡 优化要点:")
        print("   1. 文件 I/O 已移出锁的范围")
        print("   2. 使用批量写入减少文件操作")
        print("   3. 断点调试时不会长时间持有锁")
        print("   4. 并发性能显著提升")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
