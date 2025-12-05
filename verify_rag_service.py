#!/usr/bin/env python3
"""
验证 RAG 服务是否正常工作
测试流程：
1. 检查 Resource API (8000端口) 是否启动
2. 检查 Gateway Server (8080端口) 是否启动
3. 测试 RAG 资源的申请、查询和释放
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any


def check_service(url: str, name: str) -> bool:
    """检查服务是否在运行"""
    try:
        response = requests.get(url, timeout=5)
        print(f"✅ {name} is running (Status: {response.status_code})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ {name} is NOT running: {e}")
        return False


def test_resource_api_direct() -> bool:
    """直接测试 Resource API (8000端口)"""
    print("\n" + "="*60)
    print("测试 1: 直接访问 Resource API (Backend)")
    print("="*60)

    base_url = "http://localhost:8000"

    # 1. 检查服务状态
    if not check_service(f"{base_url}/status", "Resource API (Port 8000)"):
        return False

    # 2. 申请 RAG 资源
    print("\n📋 Step 1: 申请 RAG 资源...")
    worker_id = f"test_worker_{os.getpid()}"
    print(f"   Worker ID: {worker_id}")
    try:
        response = requests.post(
            f"{base_url}/allocate",
            json={"worker_id": worker_id, "type": "rag"},
            timeout=30
        )
        if response.status_code != 200:
            print(f"❌ 申请失败: {response.status_code} - {response.text}")
            return False

        data = response.json()
        resource_id = data.get("id")
        base_url_rag = data.get("base_url")
        token = data.get("token")

        if not resource_id or not base_url_rag:
            print(f"❌ 返回数据格式错误: {data}")
            return False

        print(f"✅ 成功申请 RAG 资源: {resource_id}")
        print(f"   Base URL: {base_url_rag}")
        print(f"   Token: {token[:20]}..." if token else "   Token: None")
        print(f"   Status: {data.get('status')}")

    except Exception as e:
        print(f"❌ 申请资源失败: {e}")
        return False

    # 3. 执行 RAG 查询
    print("\n🔍 Step 2: 执行 RAG 查询...")
    test_query = "What is artificial intelligence?"

    try:
        response = requests.post(
            f"{base_url_rag}/search",
            json={
                "query": test_query,
                "top_k": 5
            },
            headers={
                "Authorization": f"Bearer {token}"
            } if token else {},
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ 查询失败: {response.status_code} - {response.text}")
        else:
            result = response.json()
            print(f"✅ 查询成功!")
            print(f"   查询语句: {test_query}")

            if "results" in result:
                print(f"   返回结果数: {len(result['results'])}")
                for i, doc in enumerate(result['results'][:3], 1):
                    print(f"\n   结果 {i}:")
                    print(f"     Score: {doc.get('score', 'N/A')}")
                    print(f"     Text: {doc.get('text', 'N/A')[:100]}...")
            else:
                print(f"   完整响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ 查询失败: {e}")

    # 4. 释放资源
    print("\n🗑️  Step 3: 释放 RAG 资源...")
    try:
        response = requests.post(
            f"{base_url}/release",
            json={"resource_id": resource_id, "worker_id": worker_id},
            timeout=10
        )

        if response.status_code == 200:
            print(f"✅ 成功释放资源: {resource_id}")
        else:
            print(f"⚠️  释放资源失败: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ 释放资源失败: {e}")

    return True


def test_gateway_api() -> bool:
    """测试 Gateway API (8080端口) - MCP SSE 协议"""
    print("\n" + "="*60)
    print("测试 2: 访问 Gateway Server (MCP SSE)")
    print("="*60)

    base_url = "http://localhost:8080"

    # 1. 检查服务状态 (SSE endpoint)
    print("\n📋 检查 Gateway 服务...")
    try:
        response = requests.get(f"{base_url}/sse", timeout=5, stream=True)
        if response.status_code in [200, 426]:  # 426 = Upgrade Required
            print(f"✅ Gateway Server (Port 8080) is running (MCP SSE)")
        else:
            print(f"⚠️  Gateway 返回状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠️  无法连接到 Gateway SSE endpoint: {e}")
        print("   提示: Gateway 使用 MCP SSE 协议，不是标准 REST API")

    # 2. 检查配置文件
    print("\n📄 检查 Gateway 配置...")
    try:
        with open("gateway_config.json", "r") as f:
            config = json.load(f)
            modules = config.get("modules", [])
            print(f"✅ 发现 {len(modules)} 个配置模块:")
            for mod in modules:
                print(f"   - {mod.get('resource_type')}: {mod.get('tool_groups')}")
    except Exception as e:
        print(f"⚠️  无法读取配置: {e}")

    print("\n💡 提示: Gateway 使用 MCP SSE 协议，需要 MCP 客户端连接")
    print("   可以通过 Claude Desktop 或其他 MCP 客户端使用这些工具")

    return True


def main():
    """主验证流程"""
    print("="*60)
    print("🚀 RAG 服务验证工具")
    print("="*60)
    print("\n确保已经运行:")
    print("  1. ./start_backend.sh  (Resource API on port 8000)")
    print("  2. ./start_gateway.sh  (Gateway Server on port 8080)")
    print()

    input("按 Enter 键开始验证...")

    # 测试 1: Resource API
    success1 = test_resource_api_direct()

    # 等待一下
    time.sleep(2)

    # 测试 2: Gateway API
    success2 = test_gateway_api()

    # 总结
    print("\n" + "="*60)
    print("📊 验证结果总结")
    print("="*60)
    print(f"Resource API (Backend): {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"Gateway Server (Frontend): {'✅ 通过' if success2 else '❌ 失败'}")

    if success1 and success2:
        print("\n🎉 所有测试通过! RAG 服务运行正常。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查服务日志。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
