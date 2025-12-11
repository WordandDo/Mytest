## MCP 工具描述规范（Server 侧）

目标：让 FastMCP 生成的 `description` 和 `inputSchema` 清晰传递给客户端/LLM，减少调用歧义。

### 基本要求
- **Docstring 要写清格式/语义**：在工具函数的 docstring 中说明用途、输入格式、边界/限制（例如坐标含义、可选/必填行为、是否需要等待等）。
- **参数约束用 `Annotated` + `Field` / `Literal`**：
  - `Annotated[str, Field(description="...", min_length=1, pattern="...")]`
  - `Annotated[int, Field(ge=1, le=20)]`
  - `Literal["left", "right", "middle"]` 表达枚举。
  - 列表/嵌套对象同理，适用 `min_length`/`pattern`/`enum` 等约束。
- **worker_id 处理**：在 docstring 里声明 “worker_id 由客户端自动注入”，但仍保留参数以便 schema 生成；客户端可按需隐藏该字段。
- **隐藏工具**：对不希望暴露给 LLM 的工具使用 `hidden=True`，装饰器会自动在 docstring 前加 `[HIDDEN]`。

### 推荐写法示例
```python
from typing import Annotated, Literal
from pydantic import Field
from mcp_server.core.registry import ToolRegistry

@ToolRegistry.register_tool("desktop_action_example")
async def click_button(
    worker_id: Annotated[str, Field(description="Auto-injected by client; do not fill manually.")],
    x: Annotated[int, Field(description="Target X in pixels.", ge=0)],
    y: Annotated[int, Field(description="Target Y in pixels.", ge=0)],
    button: Annotated[Literal["left", "right", "middle"], Field(description="Mouse button to press.")] = "left",
) -> str:
    """
    Click a screen coordinate. Include small sleeps if the UI needs time to respond.
    """
    ...
```

### 校验清单
- 是否为每个参数补充了 `description`，必要时加 `min_length` / `pattern` / `enum` / `ge` / `le` 等约束。
- Docstring 是否说明了调用前置条件、自动注入字段（如 worker_id）、隐藏/公开策略。
- 工具组名称与网关配置 `tool_groups` 是否一致。
- 对返回值格式（JSON 文本、截图等）是否在 docstring 中说明。***
