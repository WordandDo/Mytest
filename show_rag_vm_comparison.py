#!/usr/bin/env python3
"""
RAG vs VM 流程对比可视化展示
"""

def print_comparison():
    print("\n" + "=" * 80)
    print("RAG 与 VM 流程对比")
    print("=" * 80)

    print("\n【统一流程】所有资源类型共享")
    print("-" * 80)

    flow = [
        ("1. 资源分配", "allocate_batch_resources", "向后端 API 请求资源"),
        ("2. 会话同步", "_sync_resource_sessions", "同步到全局会话字典"),
        ("3. 资源初始化", "{type}_initialization", "执行特定初始化逻辑"),
        ("4. 获取初始观测", "get_batch_initial_observations", "获取资源初始状态"),
    ]

    for stage, func, desc in flow:
        print(f"  {stage:15} → {func:30} # {desc}")

    print("\n【差异对比】具体实现不同")
    print("-" * 80)

    comparisons = [
        ("维度", "RAG", "VM"),
        ("-" * 20, "-" * 25, "-" * 30),
        ("资源性质", "无状态 HTTP 服务", "有状态虚拟机"),
        ("连接方式", "直连 (base_url)", "持久连接 (Controller)"),
        ("返回字段", "{id, token, base_url}", "{id, ip, port, vnc_port}"),
        ("会话存储", "RAG_SESSIONS[worker_id]", "VM_SESSIONS[worker_id]"),
        ("会话内容", "{url, token, config_top_k}", "{controller, env_id, task_id}"),
        ("初始化复杂度", "低 (配置参数)", "高 (环境准备/脚本执行)"),
        ("初始观测", "不需要", "必需 (截屏 + a11y 树)"),
        ("工具调用", "query_knowledge_base", "pyautogui_* 系列"),
        ("状态管理", "无状态", "有状态"),
    ]

    # 计算列宽
    col_widths = [
        max(len(row[0]) for row in comparisons),
        max(len(row[1]) for row in comparisons),
        max(len(row[2]) for row in comparisons),
    ]

    for row in comparisons:
        print(f"  {row[0]:{col_widths[0]}}  {row[1]:{col_widths[1]}}  {row[2]:{col_widths[2]}}")

    print("\n【关键代码位置】")
    print("-" * 80)

    locations = [
        ("RAG 会话同步", "system_tools.py:227-245", "_sync_resource_sessions"),
        ("VM 会话同步", "system_tools.py:247-286", "_sync_resource_sessions"),
        ("RAG 初始化", "rag_server.py:33-72", "rag_initialization"),
        ("VM 初始化", "vm_pyautogui_server.py:41-126", "vm_pyautogui_initialization"),
        ("RAG 查询工具", "rag_server.py:131-200", "query_knowledge_base"),
        ("VM 操作工具", "vm_pyautogui_server.py", "pyautogui_* 系列"),
    ]

    for desc, location, func in locations:
        print(f"  {desc:20} → {location:35} # {func}")

    print("\n【工具调用示例】")
    print("-" * 80)

    print("\n  RAG (无状态):")
    print("    query_knowledge_base(worker_id, query, top_k)")
    print("      ↓")
    print("    使用 session['base_url'] + session['token']")
    print("      ↓")
    print("    HTTP POST 到 RAG Service")
    print("      ↓")
    print("    返回查询结果 (JSON)")

    print("\n  VM (有状态):")
    print("    pyautogui_click(worker_id, x, y)")
    print("      ↓")
    print("    使用 session['controller']")
    print("      ↓")
    print("    RPC 调用到 VM")
    print("      ↓")
    print("    VM 执行点击，状态改变")

    print("\n【统一架构的优势】")
    print("-" * 80)

    advantages = [
        "✓ 代码复用：所有资源共享相同的分配/释放逻辑",
        "✓ 扩展性：添加新资源只需实现 {type}_initialization",
        "✓ 一致性：统一的生命周期管理和错误处理",
        "✓ 可维护性：修改基础设施自动应用到所有资源",
        "✓ 原子性：支持多资源同时分配 (RAG + VM)",
    ]

    for adv in advantages:
        print(f"  {adv}")

    print("\n【核心设计思想】")
    print("-" * 80)
    print("  统一的架构 (Architecture)  → 便于维护和扩展")
    print("  差异化的实现 (Implementation) → 满足不同资源需求")
    print("  标准化的接口 (Interface)    → {type}_initialization 模式")

    print("\n" + "=" * 80)
    print("总结：RAG 和 VM 在架构层面完全统一，在实现细节上根据资源特性灵活调整")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_comparison()
