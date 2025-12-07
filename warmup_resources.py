#!/usr/bin/env python3
"""
资源预热脚本 - 确保所有后端服务完全就绪

此脚本会：
1. 检查 Resource API 的可用性
2. 检查 RAG 服务的健康状态
3. 执行实际的 RAG 查询测试（触发索引加载）
4. 测试不同的检索模式（sparse/dense）
5. 验证响应时间和结果质量
"""

import sys
import time
import requests
import argparse
from typing import Dict, Any, Tuple
from colorama import Fore, Style, init

# 初始化 colorama（跨平台颜色支持）
init(autoreset=True)

# 配置
RESOURCE_API_URL = "http://localhost:8000"
RAG_SERVICE_URL = "http://localhost:8001"
MAX_WAIT_TIME = 600  # 最大等待时间（秒）


def print_info(msg: str):
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {msg}")


def print_success(msg: str):
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")


def print_warning(msg: str):
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")


def print_error(msg: str):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")


def check_resource_api() -> bool:
    """检查 Resource API 是否就绪"""
    print_info("Checking Resource API availability...")

    try:
        response = requests.get(f"{RESOURCE_API_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print_success(f"Resource API is available")
            print_info(f"Resource pools: {list(status.keys())}")
            return True
        else:
            print_error(f"Resource API returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to connect to Resource API: {e}")
        return False


def wait_for_rag_ready(timeout: int = MAX_WAIT_TIME) -> bool:
    """等待 RAG 服务完全就绪（索引加载完成）"""
    print_info(f"Waiting for RAG service to be ready (timeout={timeout}s)...")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=5)

            if response.status_code == 200:
                health = response.json()
                status = health.get("status")
                ready = health.get("ready")

                if status != last_status:
                    print_info(f"RAG service status: {status}, ready: {ready}")
                    last_status = status

                if ready:
                    elapsed = time.time() - start_time
                    print_success(f"RAG service is ready (took {elapsed:.1f}s)")
                    return True
            else:
                print_warning(f"Health check returned status {response.status_code}")

        except requests.exceptions.RequestException as e:
            if "Connection" in str(e) and last_status is None:
                print_info("Waiting for RAG service to start...")
                last_status = "waiting"

        time.sleep(2)

    print_error(f"RAG service did not become ready within {timeout}s")
    return False


def test_rag_query(query: str, search_type: str = "dense", top_k: int = 5) -> Tuple[bool, Dict[str, Any]]:
    """测试 RAG 查询"""
    print_info(f"Testing RAG query (search_type={search_type}, top_k={top_k})...")
    print_info(f"Query: '{query}'")

    try:
        start_time = time.time()
        response = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json={"query": query, "top_k": top_k, "search_type": search_type},
            timeout=30
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            num_results = len(result.get("results", []))

            print_success(f"Query successful (took {elapsed:.2f}s)")
            print_info(f"Retrieved {num_results} results")

            if num_results > 0:
                first_result = result["results"][0]
                print_info(f"Top result preview: {str(first_result)[:100]}...")

            return True, {"elapsed": elapsed, "num_results": num_results, "response": result}
        else:
            print_error(f"Query failed with status {response.status_code}")
            print_error(f"Response: {response.text}")
            return False, {}

    except Exception as e:
        print_error(f"Query failed with exception: {e}")
        return False, {}


def run_warmup_tests(args):
    """运行完整的预热测试套件"""
    print_success("=" * 60)
    print_success("Resource Warmup Test Suite")
    print_success("=" * 60)
    print()

    # 步骤 1: 检查 Resource API
    if not check_resource_api():
        print_error("Resource API is not available. Exiting.")
        return False

    print()

    # 步骤 2: 等待 RAG 服务就绪
    if not wait_for_rag_ready(timeout=args.timeout):
        print_error("RAG service did not become ready. Exiting.")
        return False

    print()

    # 步骤 3: 执行测试查询
    test_queries = [
        ("What is artificial intelligence?", "dense", 5),
        ("Python programming language", "dense", 3),
    ]

    # 如果启用了 hybrid 模式，也测试 sparse
    if args.test_sparse:
        test_queries.append(("machine learning algorithms", "sparse", 5))

    all_success = True
    for query, search_type, top_k in test_queries:
        success, stats = test_rag_query(query, search_type, top_k)
        if not success:
            all_success = False
        print()

    # 步骤 4: 显示总结
    print_success("=" * 60)
    if all_success:
        print_success("✅ All warmup tests passed!")
        print_success("Backend services are fully ready for use.")
    else:
        print_warning("⚠️  Some warmup tests failed.")
        print_warning("Services may not be fully ready. Check logs for details.")
    print_success("=" * 60)

    return all_success


def main():
    parser = argparse.ArgumentParser(description="Warmup backend resources")
    parser.add_argument(
        "--timeout",
        type=int,
        default=MAX_WAIT_TIME,
        help=f"Maximum wait time in seconds (default: {MAX_WAIT_TIME})"
    )
    parser.add_argument(
        "--test-sparse",
        action="store_true",
        help="Also test sparse (BM25) retrieval mode"
    )
    parser.add_argument(
        "--resource-api-url",
        default=RESOURCE_API_URL,
        help=f"Resource API URL (default: {RESOURCE_API_URL})"
    )
    parser.add_argument(
        "--rag-service-url",
        default=RAG_SERVICE_URL,
        help=f"RAG Service URL (default: {RAG_SERVICE_URL})"
    )

    args = parser.parse_args()

    # 更新全局 URL
    global RESOURCE_API_URL, RAG_SERVICE_URL
    RESOURCE_API_URL = args.resource_api_url
    RAG_SERVICE_URL = args.rag_service_url

    # 运行预热测试
    success = run_warmup_tests(args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
