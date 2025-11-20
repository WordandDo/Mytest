
# AgentFlow Environment

这个模块提供了基于继承的 `Environment` 类体系，用于管理 AgentFlow 中的工具、配置、系统提示词以及底层资源。

## 模块结构

- **核心基类**: `Environment` (`enviroment.py`) - 所有环境的抽象基类。
- **数据模型**: `data_models.py` - 定义了轨迹、观察和步骤的标准数据结构。
- **工厂模式**: `factory.py` - 统一的环境注册与创建入口。
- **内置环境**: 包含 Math, Python, Web, RAG, OSWorld, TBench 等多种实现。

## 1. 功能特性

- **继承架构**: 强制规范子类实现 `mode`, `_initialize_tools`, `run_task` 等核心接口。
- **工具管理**: 自动化管理工具注册 (`register_tool`)、Schema 生成及安全执行 (`execute_tool`)。
- **标准化交互**: 使用 `TrajectoryStep` 和 `Observation` 数据类规范交互格式。
- **资源管理**: 支持重型资源（如虚拟机、浏览器）的生命周期管理 (`setup_global_resources`, `env_start`, `env_close`)。

## 2. 快速开始

### 基本使用

```python
from envs import create_math_environment, create_environment

# 方式 1: 使用特定工厂函数
env = create_math_environment(model_name="gpt-4")

# 方式 2: 使用通用工厂 (推荐)
env = create_environment("math", model_name="gpt-4")

# 运行任务
task = {"question": "Calculate 2+2", "id": "test_01"}
result = env.run_task(task, agent_config={}, logger=None)
print(result)
````

## 3\. 内置环境类型 (Built-in Environments)

框架内置了多种环境以适应不同任务需求：

| 环境类 (`Class`) | 模式名 (`mode`) | 主要功能 | 关键工具/特性 |
| :--- | :--- | :--- | :--- |
| **MathEnvironment** | `"math"` | 数学计算 | `CalculatorTool` |
| **PythonEnvironment** | `"python"` | 代码执行 | `PythonInterpreterTool` |
| **WebEnvironment** | `"web"` | 网页浏览 | `WebSearchTool`, 浏览器控制 |
| **RAGEnvironment** | `"rag"` | 知识库检索 | `QueryRAGIndexTool` |
| **OSWorldEnvironment** | `"osworld"` | 桌面自动化 | `DesktopEnv`, `Computer13`/`PyAutoGUI` 工具集 |
| **TBenchEnvironment** | `"tbench"` | 配置基准测试 | 仅用于配置场景，无默认工具 |

## 4\. 开发指南 (Development Guide)

### 步骤 1: 继承与接口实现

```python
from envs import Environment
from envs.data_models import Observation

class MyCustomEnvironment(Environment):
    @property
    def mode(self) -> str:
        return "custom"  # 唯一标识符

    def _initialize_tools(self):
        # 注册工具
        self.register_tool(MyTool())

    def run_task(self, task, agent_config, logger) -> Dict:
        # 核心执行逻辑
        return {"answer": "result", "success": True}
```

### 步骤 2: 环境注册 (必须)

为了让 `create_environment` 工厂识别新环境，必须在模块加载时注册：

```python
from envs.factory import register_environment
register_environment("custom", MyCustomEnvironment)
```

### 步骤 3: 使用数据模型 (推荐)

建议在 `run_task` 中使用标准数据模型来记录轨迹：

```python
from envs.data_models import TrajectoryStep, Observation

# 创建观察对象
obs = Observation(type="text", content="Tool output...")
# 记录步骤
step = TrajectoryStep(step_id=1, action="tool_name", action_input={...}, observations=[obs])
```

## 5\. OSWorld 环境特别说明

`OSWorldEnvironment` 是一个复杂的桌面自动化环境，支持虚拟机控制。

### 核心配置 (`kwargs`)

  - **`provider_name`**: VM 提供商 (`vmware`, `aliyun`, `aws` 等)。
  - **`path_to_vm`**: 虚拟机镜像路径。
  - **`action_space`**: 动作空间类型 (`computer_13` 或 `pyautogui`)。
  - **`observation_type`**: 观察类型 (`screenshot`, `a11y_tree`, `screenshot_a11y_tree`)。

### 生命周期

OSWorld 需要显式的生命周期管理：

1.  **`env_start()`**: 初始化虚拟机和桌面连接。
2.  **`reset(task)`**: 重置环境状态以开始新任务。
3.  **`env_close()`**: 释放虚拟机资源。

## 6\. API 参考

### Environment 基类

  - `register_tool(tool)`: 注册工具。
  - `execute_tool(name, params)`: 安全执行工具。
  - `get_system_prompt(**kwargs)`: 获取填充后的 System Prompt。

### Factory (envs.factory)

  - `create_environment(mode, **kwargs)`: 创建环境实例。
  - `register_environment(mode, cls)`: 注册新环境类。
  - `list_registered_environments()`: 查看所有可用环境。

### Data Models (envs.data\_models)

  - **`Observation`**: `type` (text/image), `content`, `metadata`。
  - **`TrajectoryStep`**: `action`, `input`, `thought`, `observations`。
  - **`TaskTrajectory`**: 包含完整的 `steps` 列表和任务元数据。

