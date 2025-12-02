# 从核心 osworld_server 导入通用逻辑
from mcp_server.osworld_server import vm_initialization

# [关键映射]：函数名必须匹配 {res_type}_initialization
# 关键词: vm_pyautogui_initialization
vm_pyautogui_initialization = vm_initialization