import subprocess
import sys
import os
import time
import socket
import psutil  # 用于杀进程，如果没有安装，脚本有fallback
from pathlib import Path

# ================= 配置区域 =================
PYTHON_EXE = sys.executable
BACKEND_PORT = 8001
GATEWAY_PORT = 8080
DEPLOYMENT_CONFIG = "/home/a1/sdb/lb/Mytest/deployment_config_hybridrag_osworld.json"
GATEWAY_CONFIG = "gateway_config_osworld_hybirdrag.json"

# 路径配置
SEEDS_FILE = "/home/a1/sdb/lb/Mytest/src/data_synthesis/example_seed_texts.json"
OUTPUT_BASE = f"synthesis_results_simple_answer_{time.strftime('%Y%m%d_%H%M%S')}_test_seeds"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 任务定义：仅 Dense 模式
TASKS = [
    {
        "mode": "denseonly",
        "synthesis_config": "src/data_synthesis/configs/rag_config_dense.json",
        "tool_whitelist": [
            "setup_rag_session",
            "query_knowledge_base_dense"
        ]
    }
]
# ===========================================

def check_port(port, host='localhost'):
    """检查端口是否开放"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

def wait_for_port(port, name, timeout=60):
    """等待端口开放"""
    print(f"⏳ Waiting for {name} (Port {port})...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_port(port):
            print(f"✅ {name} is ready!")
            return True
        time.sleep(1)
    print(f"❌ Timeout waiting for {name}")
    return False

def kill_process_on_port(port):
    """杀掉占用指定端口的进程 (类似 lsof -ti:port | xargs kill -9)"""
    found = False
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"🛑 Killing existing process on port {port}: {proc.pid} ({proc.name()})")
                    proc.kill()
                    found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if found:
        time.sleep(2) # 等待释放

def ensure_backend():
    """确保 RAG Backend 正在运行"""
    if check_port(BACKEND_PORT):
        print("✅ Backend already running.")
        return None # 返回 None 表示我们没有启动它，所以不需要我们关闭它

    print("🚀 Starting Backend (src/mcp_server/rag_server.py)...")
    # 在后台启动
    log_file = open(LOG_DIR / "backend.log", "w")
    proc = subprocess.Popen(
        [PYTHON_EXE, "src/mcp_server/rag_server.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    if wait_for_port(BACKEND_PORT, "Backend"):
        print("⏳ Waiting 10s for index warm-up...")
        time.sleep(10)
        return proc
    else:
        print("❌ Failed to start Backend.")
        proc.kill()
        sys.exit(1)

def start_gateway(config_file):
    """启动 Gateway（混合资源网关，使用白名单在客户端侧收敛可用工具）"""
    print(f"------------------------------------------------")
    print(f"🔌 Starting Gateway with: {config_file}")
    print(f"------------------------------------------------")
    
    # 如果端口已被占用，视为已有 Gateway 运行，直接复用
    if check_port(GATEWAY_PORT):
        print("✅ Gateway already running on target port, reusing existing instance.")
        return None  # 返回 None 表示未由本脚本启动

    # 1. 清理端口（只在未运行时执行）
    kill_process_on_port(GATEWAY_PORT)
    
    # 2. 启动新 Gateway
    log_file = open(LOG_DIR / "gateway.log", "w")
    proc = subprocess.Popen(
        [PYTHON_EXE, "src/mcp_server/main.py", "--config", config_file, "--port", str(GATEWAY_PORT)],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    if wait_for_port(GATEWAY_PORT, "Gateway"):
        print("⏳ Sleeping 5s to ensure Gateway connects to RAG Server...")
        time.sleep(5)
        return proc
    else:
        print("❌ Failed to start Gateway.")
        proc.kill()
        return None

def run_synthesis(task):
    """运行数据合成 Pipeline (Multi)"""
    mode = task["mode"]
    config_path = task["synthesis_config"]
    tool_whitelist = task.get("tool_whitelist", [])
    print(f"🧠 >>> Starting Synthesis Pipeline (Multi): [{mode}] <<<")
    
    cmd = [
        PYTHON_EXE, 
        "src/data_synthesis/synthesis_pipeline_multi.py",
        "--config", config_path,
        "--seeds", SEEDS_FILE,
        "--output-dir", os.path.join(OUTPUT_BASE, mode)
    ]

    # 基于模式收敛客户端暴露的工具，确保仅检索 + 会话初始化
    env = os.environ.copy()
    env["DEPLOYMENT_CONFIG_PATH"] = DEPLOYMENT_CONFIG
    env["MCP_TOOL_WHITELIST"] = ",".join(
        t for t in tool_whitelist
    )
    
    log_path = LOG_DIR / f"synthesis_{mode}.log"
    try:
        log_file = open(log_path, "w")
        return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception as exc:
        print(f"❌ Failed to start synthesis for [{mode}]: {exc}")
        return None

def main():
    # 0. 检查必要的库
    try:
        import psutil
    except ImportError:
        print("⚠️  Installing missing dependency: psutil")
        subprocess.run([PYTHON_EXE, "-m", "pip", "install", "psutil"], check=True)
        import psutil

    # 1. 准备后端
    # 同时向子进程注入部署配置，确保资源侧使用 hybridrag osworld 配置
    os.environ["DEPLOYMENT_CONFIG_PATH"] = DEPLOYMENT_CONFIG
    backend_proc = ensure_backend()
    
    processes = []
    started_gateway = False
    try:
        # 2. 启动混合资源 Gateway（单实例复用）
        gateway_proc = start_gateway(GATEWAY_CONFIG)
        started_gateway = gateway_proc is not None
        if started_gateway is False and not check_port(GATEWAY_PORT):
            print("❌ Gateway failed to start or detect. Abort.")
            return

        # 3. 仅 Dense 任务
        for task in TASKS:
            mode = task["mode"]

            print(f"\n\n{'='*60}")
            print(f"🌊 Processing Workflow (Dense): {mode.upper()}")
            print(f"{'='*60}")

            proc = run_synthesis(task)
            if proc:
                processes.append((mode, proc))
            else:
                print(f"❌ Failed to launch process for {mode}")

        # 4. 等待任务结束
        for mode, proc in processes:
            ret = proc.wait()
            if ret == 0:
                print(f"✅ Synthesis for [{mode}] completed.")
            else:
                print(f"❌ Synthesis for [{mode}] failed with code {ret}.")

        # 5. 关闭 Gateway（仅当本脚本启动时）
        if started_gateway and gateway_proc:
            print(f"🛑 Stopping shared Gateway...")
            gateway_proc.terminate()
            gateway_proc.wait()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted, terminating child processes...")
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
    finally:
        # 脚本退出时的清理
        print("\n🧹 Final Cleanup...")
        # 如果是我们启动的后端，则关闭它；如果是已经存在的，则保留
        if backend_proc:
            print("🛑 Stopping Backend...")
            backend_proc.terminate()
        # 仅在我们启动 Gateway 时才清理端口，避免误杀外部实例
        if started_gateway:
            kill_process_on_port(GATEWAY_PORT)
        print("🎉 Done!")

if __name__ == "__main__":
    main()
