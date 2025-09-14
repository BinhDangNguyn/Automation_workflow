

import requests
import json
import sys
import inspect
import importlib
import logging
import traceback
import os

# =========================
# Logger setup
# =========================
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'D:/Python3.10/Code/LLM/chatbot_debug.log'
if os.path.exists(log_file):
    os.remove(log_file)

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


# =========================
# Tool Manager
# =========================
class ToolManager:
    def __init__(self, tools_module_name):
        self.module = importlib.import_module(tools_module_name)
        self.schemas = self._build_tools_schema()

    def _build_tools_schema(self):
        schemas = []
        for name, func in inspect.getmembers(self.module, inspect.isfunction):
            if name.startswith("_"):
                continue

            description = (func.__doc__ or "").strip()
            sig = inspect.signature(func)
            params = {"type": "object", "properties": {}, "required": []}

            for param in sig.parameters.values():
                params["properties"][param.name] = {"type": "string"}
                params["required"].append(param.name)

            if not params["properties"]:
                params = {"type": "object", "properties": {}}

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params
                }
            }
            schemas.append(schema)
        return schemas

    def get_function(self, name):
        return getattr(self.module, name, None)

    def get_schemas(self):
        return self.schemas


# =========================
# Memory Buffer
# =========================
class MemoryBuffer:
    def __init__(self, system_prompt=""):
        self.history = []
        if system_prompt:
            self.add_message("system", system_prompt)

    def add_message(self, role, content, **kwargs):
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.history.append(msg)

    def get_history(self):
        return self.history

    def replace_history(self, new_history):
        self.history = new_history


# =========================
# LLM ChatBot
# =========================
class LLMChatBot:
    def __init__(self, model_name, api_url, memory_buffer, tool_manager=None):
        self.model_name = model_name
        self.api_url = api_url
        self.tool_manager = tool_manager
        self.memory = memory_buffer
        self.save_values = {}

    def _call_llm(self):
        messages = self.memory.get_history()
        logger.debug("=== Messages gửi lên model ===\n%s", json.dumps(messages, ensure_ascii=False, indent=2))

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "tools": self.tool_manager.get_schemas(),
                    "stream": False,
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("=== Raw result từ model ===\n%s", json.dumps(result, ensure_ascii=False, indent=2))

            message = result.get("message", {})
            self.memory.add_message(message.get("role"), message.get("content", ""))

            # In ra màn hình nếu model trả lời
            if message.get("role") == "assistant" and message.get("content"):
                print(f"LLM: {message['content']}")

            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                self._handle_tool_calls(tool_calls)

        except Exception as e:
            logger.error(f"Lỗi khi gọi LLM: {e}")
            traceback.print_exc()

    # def _call_llm(self):
    #     messages = self.memory.get_history()
    #     logger.debug("=== Messages gửi lên model ===\n%s", json.dumps(messages, ensure_ascii=False, indent=2))

    #     try:
    #         response = requests.post(
    #             self.api_url,
    #             json={
    #                 "model": self.model_name,
    #                 "messages": messages,
    #                 "tools": self.tool_manager.get_schemas(),
    #                 "stream": False,
    #             },
    #             headers={"Content-Type": "application/json"}
    #         )
    #         response.raise_for_status()
    #         result = response.json()
    #         logger.debug("=== Raw result từ model ===\n%s", json.dumps(result, ensure_ascii=False, indent=2))

    #         message = result.get("message", {})
    #         self.memory.add_message(message.get("role"), message.get("content", ""))

    #         tool_calls = message.get("tool_calls", [])
    #         if tool_calls:
    #             self._handle_tool_calls(tool_calls)

    #     except Exception as e:
    #         logger.error(f"Lỗi khi gọi LLM: {e}")
    #         traceback.print_exc()

    def _handle_tool_calls(self, tool_calls):
        for call in tool_calls:
            func_info = call.get("function", {})
            tool_name = func_info.get("name")
            args = func_info.get("arguments", {})

            func = self.tool_manager.get_function(tool_name)
            if not func:
                tool_result = {"error": f"Không tìm thấy tool {tool_name}"}
            else:
                try:
                    if tool_name != "finish_session":
                        self.save_values.update(args)
                    if tool_name == "finish_session":
                        tool_result = func(**self.save_values)
                    else:
                        tool_result = func(**args) if args else func()
                except Exception as e:
                    tool_result = {"error": str(e)}

            # Lưu vào bộ nhớ
            self.memory.add_message("tool", json.dumps(tool_result, ensure_ascii=False),
                                    tool_call_id=call.get("id"), name=tool_name)

            # In kết quả tool ra màn hình
            print(f"[Tool {tool_name} Output]: {tool_result}")

        # Gọi lại LLM
        self._call_llm()
    # def _handle_tool_calls(self, tool_calls):
    #     for call in tool_calls:
    #         func_info = call.get("function", {})
    #         tool_name = func_info.get("name")
    #         args = func_info.get("arguments", {})

    #         func = self.tool_manager.get_function(tool_name)
    #         if not func:
    #             tool_result = {"error": f"Không tìm thấy tool {tool_name}"}
    #         else:
    #             try:
    #                 if tool_name != "finish_session":
    #                     self.save_values.update(args)
    #                 if tool_name == "finish_session":
    #                     tool_result = func(**self.save_values)
    #                 else:
    #                     tool_result = func(**args) if args else func()
    #             except Exception as e:
    #                 tool_result = {"error": str(e)}

    #         self.memory.add_message("tool", json.dumps(tool_result, ensure_ascii=False),
    #                                 tool_call_id=call.get("id"), name=tool_name)

    #     self._call_llm()  # Gọi lại LLM sau khi có tool output

    def chat(self):
        print("=== Bắt đầu chat với LLM (gõ 'exit' để thoát) ===")
        while True:
            user_input = input("Bạn: ")
            if user_input.lower() in ["exit", "quit", "thoát"]:
                break
            self.memory.add_message("user", user_input)
            self._call_llm()
    
    def ask_once(self, user_input: str) -> str:
        """Hàm gọi 1 câu hỏi, trả về string kết quả từ LLM"""
        self.memory.add_message("user", user_input)
        messages = self.memory.get_history()
        logger.debug("=== Messages gửi lên model ===\n%s", json.dumps(messages, ensure_ascii=False, indent=2))

        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            }
            
            # Only add tools if tool_manager exists (for tool-capable models)
            if hasattr(self, 'tool_manager') and self.tool_manager is not None:
                payload["tools"] = self.tool_manager.get_schemas()
                logger.debug("Including tools in request for tool-capable model")
            else:
                logger.debug("No tools included - model doesn't support tool calling")
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Add timeout for slow models
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("=== Raw result từ model ===\n%s", json.dumps(result, ensure_ascii=False, indent=2))

            message = result.get("message", {})
            self.memory.add_message(message.get("role"), message.get("content", ""))

            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]   # ✅ trả kết quả ra ngoài
            else:
                return ""
        except Exception as e:
            logger.error(f"Lỗi khi gọi LLM: {e}")
            traceback.print_exc()
            return ""



# =========================
# Run Example
# =========================
if __name__ == "__main__":
    tool_manager = ToolManager("tools_new")
    # f1 = open('D:/Python3.10/Code/LLM_with_tools/New_CSKH.txt', 'r', encoding='utf-8')
    # system_prompt = f1.read().strip()
    # f1.close()
    system_prompt = "Bạn là một trợ lý ảo chuyên nghiệp, hãy giúp tôi trả lời các câu hỏi của khách hàng một cách nhanh chóng và chính xác. Nếu cần sử dụng công cụ, hãy gọi đúng tên và truyền tham số chính xác."

    memory = MemoryBuffer(system_prompt)
    bot = LLMChatBot(
        model_name="qwen3:8b",
        api_url="http://localhost:11434/api/chat",
        memory_buffer=memory,
        tool_manager=tool_manager
    )
    bot.chat()

