import asyncio
import json
import sys
import os

# 确保可以将 src 目录加入路径以导入项目模块
sys.path.append(os.getcwd())

try:
    from src.utils.mcp_sse_client import MCPSSEClient
except ImportError:
    print("❌ Error: 无法导入 MCPSSEClient。请确保您在项目根目录下运行此脚本。")
    sys.exit(1)

async def main():
    # =========================================================================
    # 配置信息
    # =========================================================================
    server_url = "http://localhost:8080"  # 请确保 MCP Server 已启动
    worker_id = "test_worker_hybrid_001"  # 用于标识当前会话的 Worker ID
    
    print(f"🔌 连接到 MCP Server: {server_url} ...")
    
    # 使用 Context Manager 自动处理连接和关闭
    async with MCPSSEClient(f"{server_url}/sse") as client:
        print("✅ 已连接")

        # =========================================================================
        # 0. 资源分配 (Resource Allocation)
        # =========================================================================
        # 注意：RAG 服务通常需要先为 worker_id 分配资源。
        # 如果您的环境没有自动分配，可能需要调用 'allocate_batch_resources'。
        # 这里为了演示工具调用，假设资源已就绪或通过 Agent 环境自动管理。
        # 如果收到 "No active RAG session" 错误，请检查资源分配逻辑。
        
        try:
            print(f"\n🛠️  尝试为 {worker_id} 分配 RAG 资源...")
            # 尝试调用系统分配工具（参数根据实际 system_tools 定义可能有所不同）
            await client.call_tool("allocate_batch_resources", {
                "worker_id": worker_id, 
                "resource_types": ["rag"]
            })
            print("   资源分配请求已发送")
        except Exception as e:
            print(f"⚠️  资源分配跳过或失败 (可能是隐式分配): {e}")

        # =========================================================================
        # 1. 测试 Dense 检索 (rag_query)
        # =========================================================================
        print("\n" + "="*50)
        print("🔍 测试 1: Dense 检索 (语义搜索)")
        print("="*50)
        
        tool_dense = "rag_query"
        query_dense = "深度学习中的 Transformer 是什么？"  # 这是一个概念性问题，适合语义检索
        args_dense = {
            "worker_id": worker_id,
            "query": query_dense,
            "top_k": 3
        }
        
        print(f"调用工具: {tool_dense}")
        print(f"参数: {json.dumps(args_dense, ensure_ascii=False)}")
        
        try:
            result_dense = await client.call_tool(tool_dense, args_dense)
            _print_result(result_dense)
        except Exception as e:
            print(f"❌ 调用失败: {e}")

        # =========================================================================
        # 2. 测试 Sparse 检索 (rag_query_sparse)
        # =========================================================================
        print("\n" + "="*50)
        print("🔍 测试 2: Sparse 检索 (关键词匹配)")
        print("="*50)
        
        tool_sparse = "rag_query_sparse"
        query_sparse = "BERT model_id: 1024"  # 这是一个包含特定 ID 或术语的查询，适合关键词检索
        args_sparse = {
            "worker_id": worker_id,
            "query": query_sparse,
            "top_k": 3
        }
        
        print(f"调用工具: {tool_sparse}")
        print(f"参数: {json.dumps(args_sparse, ensure_ascii=False)}")
        
        try:
            result_sparse = await client.call_tool(tool_sparse, args_sparse)
            _print_result(result_sparse)
        except Exception as e:
            print(f"❌ 调用失败: {e}")

def _print_result(result):
    """辅助函数：美化打印结果"""
    if hasattr(result, 'content') and result.content:
        for item in result.content:
            if item.type == 'text':
                try:
                    # 尝试解析 JSON 字符串
                    res_json = json.loads(item.text)
                    if res_json.get("status") == "success":
                        print("\n📄 检索结果:")
                        # RAG 返回的 results 通常是一个字符串，可能包含换行
                        print(res_json.get("results", "No content"))
                    else:
                        print(f"\n⚠️  服务端返回错误: {res_json.get('message')}")
                except:
                    print(f"\n📄 原始返回: {item.text}")
    else:
        print(f"\n📄 Result Object: {result}")

if __name__ == "__main__":
    asyncio.run(main())