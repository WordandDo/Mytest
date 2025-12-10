import subprocess
import sys
import os
import time
import socket
import signal
import psutil  # 用于杀进程，如果没有安装，脚本有fallback

# ================= 配置区域 =================
PYTHON_EXE = sys.executable
BACKEND_PORT = 8001
GATEWAY_PORT = 8080

# 路径配置
SEEDS_FILE = "/home/a1/sdb/lb/Mytest/src/data_synthesis/sample_entities_500.json"
OUTPUT_BASE = f"synthesis_results_simple_answer_{time.strftime('%Y%m%d_%H%M%S')}"

# 任务定义：(模式名称, Gateway配置文件, Synthesis配置文件)
TASKS = [
    (
        "rag_hybrid", 
        "gateway_config_rag_hybrid.json", 
        "src/data_synthesis/configs/rag_config_hybrid.json"
    ),
    (
        "rag_dense", 
        "gateway_config_rag_dense_only.json", 
        "src/data_synthesis/configs/rag_config_dense.json"
    )
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
    log_file = open("backend.log", "w")
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
    """启动 Gateway"""
    print(f"------------------------------------------------")
    print(f"🔌 Starting Gateway with: {config_file}")
    print(f"------------------------------------------------")
    
    # 1. 清理端口
    kill_process_on_port(GATEWAY_PORT)
    
    # 2. 启动新 Gateway
    log_file = open("gateway.log", "w")
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

def run_synthesis(mode, config_path):
    """运行数据合成 Pipeline (Multi)"""
    print(f"🧠 >>> Starting Synthesis Pipeline (Multi): [{mode}] <<<")
    
    cmd = [
        PYTHON_EXE, 
        "src/data_synthesis/synthesis_pipeline_multi.py",
        "--config", config_path,
        "--seeds", SEEDS_FILE,
        "--output-dir", os.path.join(OUTPUT_BASE, mode)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Synthesis for [{mode}] completed.")
    except subprocess.CalledProcessError:
        print(f"❌ Synthesis for [{mode}] failed.")

def main():
    # 0. 检查必要的库
    try:
        import psutil
    except ImportError:
        print("⚠️  Installing missing dependency: psutil")
        subprocess.run([PYTHON_EXE, "-m", "pip", "install", "psutil"], check=True)
        import psutil

    # 1. 准备后端
    backend_proc = ensure_backend()
    
    try:
        # 2. 循环执行任务
        for mode, gateway_conf, rag_conf in TASKS:
            print(f"\n\n{'='*60}")
            print(f"🌊 Processing Workflow: {mode.upper()}")
            print(f"{'='*60}")
            
            # 启动对应的 Gateway
            gateway_proc = start_gateway(gateway_conf)
            
            if gateway_proc:
                try:
                    # 运行合成
                    run_synthesis(mode, rag_conf)
                finally:
                    # 任务结束后关闭当前 Gateway，为下一次腾出端口
                    print(f"🛑 Stopping Gateway for {mode}...")
                    gateway_proc.terminate()
                    gateway_proc.wait()
            
    finally:
        # 脚本退出时的清理
        print("\n🧹 Final Cleanup...")
        # 如果是我们启动的后端，则关闭它；如果是已经存在的，则保留
        if backend_proc:
            print("🛑 Stopping Backend...")
            backend_proc.terminate()
        
        # 双重保险：清理残留的 Gateway
        kill_process_on_port(GATEWAY_PORT)
        print("🎉 Done!")

if __name__ == "__main__":
    main()