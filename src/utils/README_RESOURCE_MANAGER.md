# Resource Manager 模块
该模块实现了一个线程安全的单例资源管理器
这是一个对 `src/utils/resource_manager.py` 的深度分析与开发文档。这个模块在 AgentFlow 框架中扮演着核心角色，特别是对于需要管理重型资源（如虚拟机）的复杂环境（如 OSWorld）。

### 1\. 模块原理分析

`resource_manager.py` 的核心目标是**解决多进程/并发环境下的重资源（Heavy Resource）管理问题**。在传统的单进程脚本中，环境（Env）通常直接持有资源（如一个 `DesktopEnv` 实例对应一个 VM）。但在大规模评估或数据合成中，我们需要并行运行多个 Worker，这就带来了以下挑战：

  * **资源隔离**：每个 Worker 需要独占一个 VM，不能冲突。
  * **状态一致性**：VM 的分配、释放、重置状态需要在所有进程间同步。
  * **生命周期管理**：VM 启动慢、销毁慢，需要池化管理以复用，避免频繁创建销毁。
  * **跨进程通信**：主进程管理资源池，Worker 进程申请资源，需要高效的 IPC 机制。

该模块采用**分层架构 + 代理模式**来解决上述问题：

#### 架构分层

1.  **接口层 (`ResourceManager` / `HeavyResourceManager`)**:

      * 定义了所有资源管理器的统一契约：`initialize`, `allocate`, `release`, `stop_all`。
      * 上层业务代码（如 `ParallelOSWorldRolloutEnvironment`）只依赖此接口，不关心底层是本地 VM 还是云端实例。

2.  **代理层 (`VMPoolResourceManager`)**:

      * 这是 `HeavyResourceManager` 的具体实现。
      * 它并不直接管理 VM 的元数据，而是作为一个**外观（Facade）**。
      * **关键职责**：
          * 利用 Python `multiprocessing.managers.BaseManager` 启动一个独立的守护进程。
          * 在该守护进程中实例化真正的管理者 `VMPoolManager`。
          * 向 Worker 暴露 `VMPoolManager` 的代理对象，拦截 `allocate` 请求。
          * **Attach 模式实例化**：拿到 VM 连接信息（IP/Port）后，在 Worker 本地实例化 `DesktopEnv` 并连接到远程 VM，而不是在资源管理器进程中创建 `DesktopEnv`。这避免了复杂的对象序列化问题。

3.  **核心逻辑层 (`VMPoolManager`)**:

      * 运行在独立的守护进程中，拥有全局唯一的 VM 状态视图。
      * 维护 `vm_pool` 字典和 `free_vm_queue` 队列。
      * 处理并发锁 (`pool_lock`)，确保 `allocate_vm` 操作的原子性。
      * 直接调用底层 Provider（如阿里云、VMware 接口）控制 VM 开关机和快照恢复。

4.  **底层实现层 (`VMPoolConfig` / `VMPoolEntry`)**:

      * 定义了标准化的配置结构和资源状态数据类。

### 2\. 开发文档

以下是整理好的开发文档，建议保存为 `src/utils/README_RESOURCE_MANAGER.md` 或整合进项目文档。

-----

# Resource Manager 模块开发文档

`src/utils/resource_manager.py` 提供了一套统一的资源管理框架，用于在多进程环境下高效、安全地管理重型资源（主要是虚拟机 VM）。

## 核心概念

  * **ResourceManager**: 抽象基类，定义资源申请与释放的标准接口。
  * **VMPoolResourceManager**: 面向用户的入口类。它会自动启动一个后台进程来托管资源池，并处理跨进程通信。
  * **VMPoolManager**: 实际的资源管理者（运行在后台进程），负责维护 VM 状态、调用云厂商 API、管理快照重置。
  * **Attach Mode**: 一种设计模式。资源管理器只返回 VM 的连接信息（IP/Port），由 Worker 进程在本地创建 `DesktopEnv` 并“连接”到该 VM。

## 架构图解

```mermaid
graph TD
    subgraph MainProcess
        User[User/Runner] -->|1. create| RM[VMPoolResourceManager]
        RM -->|2. start process| BM[BaseManager]
    end

    subgraph ManagerProcess [Daemon Process]
        BM -->|hosts| PM[VMPoolManager]
        PM -->|manages| Pool[VM Pool Dict]
        PM -->|calls| Provider[Cloud/VM Provider API]
    end

    subgraph WorkerProcess
        Worker -->|3. allocate()| RM_Proxy[RM Proxy]
        RM_Proxy -->|IPC| PM
        PM -->|4. return IP/Port| RM_Proxy
        RM_Proxy -->|5. init| Env[DesktopEnv (Local)]
        Env -->|6. connect| VM[Remote VM]
    end
```

## 快速开始

### 1\. 初始化资源管理器

通常在主进程（Runner）开始时初始化。

```python
from utils.resource_manager import VMPoolResourceManager

# 配置 VM 池
pool_config = {
    "num_vms": 4,                      # 启动 4 台 VM
    "provider_name": "aliyun",         # 使用阿里云
    "path_to_vm": "img-ubuntu-base",   # 镜像 ID 或路径
    "snapshot_name": "init_state",     # 重置时的快照点
    "screen_size": (1920, 1080),
    "headless": True
}

# 创建并初始化（这会启动后台进程并等待所有 VM 就绪）
resource_manager = VMPoolResourceManager(pool_config=pool_config)
resource_manager.initialize()
```

### 2\. 在 Worker 中使用

将 `resource_manager` 传递给 Worker 进程（由于使用了 `BaseManager`，它是可被 pickle 序列化的代理对象）。

```python
def worker_task(resource_manager, worker_id):
    # 1. 申请资源 (阻塞等待直到有空闲 VM)
    # 返回的 desktop_env 已经连接到了分配的 VM
    vm_id, desktop_env = resource_manager.allocate(
        worker_id=worker_id,
        timeout=600,
        # 可以在此覆盖部分 Env 配置
        action_space="pyautogui"
    )
    
    try:
        # 2. 使用环境
        obs = desktop_env.reset(task_config={...})
        # ... 执行任务 ...
        
    finally:
        # 3. 释放资源 (自动重置 VM 状态以便复用)
        # 注意：不要直接 close desktop_env，交给 release 处理
        resource_manager.release(vm_id, worker_id, reset=True)

# 在主进程停止所有资源
resource_manager.stop_all()
```

## API 参考

### `VMPoolResourceManager`

继承自 `HeavyResourceManager`。

  * **`__init__(pool_config: Dict)`**: 构造函数。
      * `pool_config`: 包含 `num_vms`, `provider_name` 等配置。
  * **`initialize() -> bool`**: 启动 VM 池。会并发调用 Provider API 启动所有 VM。成功返回 `True`。
  * **`allocate(worker_id, timeout, **kwargs) -> (vm_id, desktop_env)`**:
      * `worker_id`: 申请者的唯一标识（用于追踪）。
      * `timeout`: 等待可用 VM 的超时时间（秒）。
      * `**kwargs`: 传递给 `DesktopEnv` 的额外初始化参数。
      * **返回**: VM ID 和初始化好的 `DesktopEnv` 实例。
  * **`release(resource_id, worker_id, reset=True)`**:
      * 归还 VM。如果 `reset=True`，后台会立即调用 Provider 恢复 VM 快照，之后该 VM 才会变回 `FREE` 状态供他人使用。
  * **`stop_all()`**: 停止所有 VM 实例并关闭管理进程。

### 配置参数 (`pool_config`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `num_vms` | int | 1 | 资源池中的 VM 数量 |
| `provider_name` | str | "vmware" | "aliyun", "aws", "vmware", "virtualbox" |
| `path_to_vm` | str | None | VM 镜像路径或云端 Image ID |
| `snapshot_name` | str | "init\_state" | 任务结束后恢复的快照名称 |
| `headless` | bool | False | 是否无头模式运行（无 GUI 窗口） |
| `action_space` | str | "computer\_13" | 动作空间类型 |

## 扩展指南

### 添加新的资源类型

如果需要管理 GPU 或 API Key 池，可以继承 `ResourceManager`：

1.  **定义接口**: 继承 `ResourceManager`。
2.  **实现状态管理**: 参考 `VMPoolManager`，使用 `Queue` 管理空闲资源。
3.  **处理并发**: 如果涉及多进程，建议同样使用 `BaseManager` 模式进行封装。

<!-- end list -->

```python
class GPUResourceManager(ResourceManager):
    # ... 实现 initialize, allocate, release ...
    pass
```

## 常见问题

1.  **为什么 `allocate` 返回的是 `DesktopEnv` 对象？**

      * 为了方便上层使用。`VMPoolManager` 只管理 IP/Port 数据，而 `VMPoolResourceManager` 负责将这些数据“组装”成可用的 `DesktopEnv` 对象。

2.  **VM 重置失败怎么办？**

      * `release` 中的重置是异步的（在后台进程执行）。如果重置失败，该 VM 会被标记为 `ERROR` 状态，不再分配给新任务，防止污染后续评估。

3.  **如何调试？**

      * 查看日志中的 `VMPoolManager` 输出。它详细记录了每个 VM 的初始化、分配、释放和重置状态。
      * 调用 `resource_manager.get_status()` 可以获取当前池的实时统计信息（空闲数、占用数、错误数）。