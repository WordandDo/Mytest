## MCP 工具描述与参数指南

### 1. 概览
MCP 工具的元数据（名称、描述、参数结构）是由服务端注册时生成，客户端通过 `tools/list` 直接拉取。你在服务端传给 FastMCP 的内容会全量透传给 Agent/OpenAI SDK，因此只要在这一步做好规范就能控制模型看到的「函数签名」。

### 2. 服务器侧：如何生成描述
- 默认行为是提取目标函数的 `docstring` 作为 `Tool.description`，也会自动使用函数名、type hints 生成 `inputSchema`（参考：`src/mcp_server/main.py` 的 `ToolRegistry` 和 FastMCP 注册流程）。
- 想要覆盖默认描述，可以在注册时显式指定 `description`、`title` 或 `annotations`：
  ```python
  @mcp.tool(
      name="count_php_lines",
      description="扫描目录并汇报 PHP 文件的总行数。",
      title="PHP 行数统计",
      annotations=ToolAnnotations(
          title="计数 PHP 文件行数",
          readOnlyHint=True,
      ),
  )
  def count_php_lines(root_path: str) -> str:
      """递归统计 PHP 文件行数，用于审计。"""
      ...
  ```
- `description` 覆盖 docstring，`title`/`annotations.title` 提供更友好的 UI/Agent 文字，而 Agent 端会根据实际实现选择优先级（例如 `HttpMCPEnv` 直接串 `Tool.description` 到 `tool_descriptions`）。
- `[HIDDEN]` 前缀可用于标记不希望开放给模型的工具（`HttpMCPEnv` 中会过滤，参见 `src/envs/http_mcp_env.py` 的白/黑名单策略）。

### 3. 客户端：如何读取并转成 OpenAI 风格描述
- `MCPSSEClient.list_tools()`（`src/utils/mcp_sse_client.py`）调用 `ClientSession.list_tools()`（`mcp.client.session.ClientSession`），拿到的 `Tool` 就包含 `name`/`description`/`inputSchema`。
- 在 `HttpMCPEnv` 里的 `_initialize_tools()` 会：
  1. 过滤白/黑名单和 `[HIDDEN]`；
  2. 把 `inputSchema` 转换成 `ToolMetadata` 并缓存；
  3. 调用 `_convert_mcp_tool_to_openai()` 生成 OpenAI SDK 期待的 `parameters`；
  4. 拼出 `tool_descriptions` 供 system prompt 注入（格式如 `- tool_name: description`）。
- 只要服务端文本写得清楚、结构完整，Agent 就能收到与 OpenAI 函数调用兼容的 metadata。

### 4. 参数约束与格式提示
- `Tool.parameters`（内部的 JSON Schema）来自 FastMCP 的 `func_metadata`，它会根据类型注解、`typing.Annotated`、pydantic/Field 信息自动生成，包含 `properties`、`required`、`pattern`、`description` 等。
- 你可以通过 `Annotated` + `Field` 强化约束，例如要求字符串符合 `(x,y)` 模式、只能是某些语言、或枚举值：
  ```python
  from typing import Annotated
  from pydantic import Field

  @mcp.tool(name="parse_coordinate")
  def parse_coordinate(
      raw: Annotated[
          str,
          Field(
              description="格式 (x,y)，x 与 y 为整数",
              regex=r"^\(\d+,\d+\)$"
          )
      ],
      language: Annotated[
          str,
          Field(
              description="支持的语言（如 python 或 bash）",
              regex="^(python|bash)$"
          )
      ]
  ) -> tuple[int, int]:
      ...
  ```
- 这样生成的 schema 就会带上 `pattern` 与详细描述，Agent 在 OpenAI 或其他客户端看到 `parameters` 时会得知不仅要 `type=string`，还需要匹配具体格式。
- 复杂结构（列表、嵌套对象、分支约束）同样可以用 `Annotated[list[str], Field(min_items=2)]`、`Literal`、`pydantic.BaseModel` 等方式表达，只要 FastMCP 能推断出 JSON Schema，客户端就可以在 `parameters` 里看到对应字段。

### 5. 小结
1. 在服务端用 docstring/显式 `description` 填充工具的文字说明，必要时通过 `title` 或 `annotations` 提供 UI-friendly 文案。
2. 依赖 FastMCP 生成的 JSON Schema 来定义参数结构，`Field`、`Annotated`、`Literal`、`pattern` 等都能传递给 Agent。
3. 客户端（如 `HttpMCPEnv`）负责拉 `tools/list`、黑白名单过滤、转成 OpenAI-style `type/function/parameters` 格式，LLM 层就能像调用标准 SDK 那样使用这些工具。
