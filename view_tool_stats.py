#!/usr/bin/env python3
"""
工具调用统计报告查看器
用于查看、导出和分析工具调用统计数据
"""
import argparse
import json
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server.core.tool_stats import get_stats_collector


def view_summary(stats_dir: str):
    """查看统计摘要"""
    collector = get_stats_collector(stats_dir)
    collector.print_summary()


def view_task(stats_dir: str, task_id: str):
    """查看特定任务的统计"""
    collector = get_stats_collector(stats_dir)
    report = collector.get_task_report(task_id)

    print(f"\n{'='*60}")
    print(f"📋 Task Report: {task_id}")
    print(f"{'='*60}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()


def view_tool(stats_dir: str, tool_name: str = None):
    """查看工具级别统计"""
    collector = get_stats_collector(stats_dir)
    report = collector.get_tool_report()

    if tool_name:
        # 筛选特定工具
        tool_data = next(
            (t for t in report["tools"] if t["tool_name"] == tool_name),
            None
        )
        if tool_data:
            print(f"\n{'='*60}")
            print(f"🔧 Tool Report: {tool_name}")
            print(f"{'='*60}")
            print(json.dumps(tool_data, indent=2, ensure_ascii=False))
        else:
            print(f"Tool '{tool_name}' not found")
    else:
        print(f"\n{'='*60}")
        print(f"🔧 All Tools Report")
        print(f"{'='*60}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    print()


def view_failures(stats_dir: str, task_id: str = None):
    """查看失败的调用"""
    collector = get_stats_collector(stats_dir)
    failed_calls = collector.get_failed_calls(task_id)

    print(f"\n{'='*60}")
    print(f"❌ Failed Calls" + (f" for Task: {task_id}" if task_id else ""))
    print(f"{'='*60}")

    if not failed_calls:
        print("No failed calls found")
    else:
        print(json.dumps(failed_calls, indent=2, ensure_ascii=False))
    print()


def export_report(stats_dir: str, output_file: str = None):
    """导出完整报告"""
    collector = get_stats_collector(stats_dir)
    filepath = collector.export_report(output_file)
    print(f"✅ Report exported to: {filepath}")


def list_tasks(stats_dir: str):
    """列出所有任务"""
    collector = get_stats_collector(stats_dir)
    report = collector.get_all_tasks_report()

    print(f"\n{'='*60}")
    print(f"📋 All Tasks ({report['summary']['total_tasks']} tasks)")
    print(f"{'='*60}\n")

    for task in report['tasks']:
        success_indicator = "✓" if task['success_rate'] >= 80 else "⚠" if task['success_rate'] >= 50 else "✗"
        print(f"{success_indicator} {task['task_id']}")
        print(f"  Calls: {task['total_calls']}, "
              f"Success: {task['success_calls']}, "
              f"Failed: {task['failed_calls']}, "
              f"Rate: {task['success_rate']}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze tool call statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary
  python view_tool_stats.py summary

  # List all tasks
  python view_tool_stats.py list-tasks

  # View specific task
  python view_tool_stats.py task --task-id task_001

  # View tool statistics
  python view_tool_stats.py tool --tool-name search_documents

  # View all failures
  python view_tool_stats.py failures

  # Export report
  python view_tool_stats.py export --output my_report.json
        """
    )

    parser.add_argument(
        "--stats-dir",
        type=str,
        default="tool_stats",
        help="Statistics directory (default: tool_stats)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Summary command
    subparsers.add_parser("summary", help="View statistics summary")

    # List tasks command
    subparsers.add_parser("list-tasks", help="List all tasks")

    # Task command
    task_parser = subparsers.add_parser("task", help="View specific task statistics")
    task_parser.add_argument("--task-id", required=True, help="Task ID")

    # Tool command
    tool_parser = subparsers.add_parser("tool", help="View tool statistics")
    tool_parser.add_argument("--tool-name", help="Specific tool name (optional)")

    # Failures command
    failures_parser = subparsers.add_parser("failures", help="View failed calls")
    failures_parser.add_argument("--task-id", help="Filter by task ID (optional)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export full report")
    export_parser.add_argument("--output", help="Output file name (optional)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "summary":
            view_summary(args.stats_dir)
        elif args.command == "list-tasks":
            list_tasks(args.stats_dir)
        elif args.command == "task":
            view_task(args.stats_dir, args.task_id)
        elif args.command == "tool":
            view_tool(args.stats_dir, getattr(args, "tool_name", None))
        elif args.command == "failures":
            view_failures(args.stats_dir, getattr(args, "task_id", None))
        elif args.command == "export":
            export_report(args.stats_dir, getattr(args, "output", None))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
