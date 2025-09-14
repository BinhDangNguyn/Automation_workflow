# node_sandbox.py
import io
import sys
import json
from contextlib import redirect_stdout
from node_library import call_llm_agent

def execute_python_node(code: str, input_data: dict) -> dict:
    """
    Thực thi code Python của một node trong môi trường được kiểm soát.
    Cảnh báo: Việc sử dụng exec() có rủi ro bảo mật. Trong môi trường production,
    hãy thay thế bằng các giải pháp sandbox mạnh mẽ hơn như Docker, gVisor, hoặc RestrictedPython.
    """
    print(f"[SANDBOX] Starting node execution")
    print(f"[SANDBOX] Input data: {input_data}")
    print(f"[SANDBOX] Code preview: {code[:100]}...")
    
    logs = []
    def console_log(message):
        msg = str(message)
        logs.append(msg)
        print(f"[NODE-LOG] {msg}")

    # Tạo một scope (môi trường) riêng cho mỗi lần thực thi
    execution_scope = {
        'input_data': input_data,
        'console_log': console_log,
        'json': json,
        # Đưa các hàm helper cần thiết vào scope
        'call_llm_agent': call_llm_agent
    }

    print(f"[SANDBOX] Execution scope keys: {list(execution_scope.keys())}")

    try:
        print(f"[SANDBOX] Compiling and executing code...")
        
        # Sử dụng redirect_stdout để bắt các lệnh print()
        f = io.StringIO()
        with redirect_stdout(f):
            # Biên dịch và thực thi code để định nghĩa hàm execute
            exec(code, execution_scope)
        
        print(f"[SANDBOX] Code compiled successfully")
        print(f"[SANDBOX] Execution scope now has: {list(execution_scope.keys())}")
        
        # Lấy lại các log từ lệnh print
        print_logs = f.getvalue()
        if print_logs:
            print_log_lines = print_logs.strip().split('\\n')
            logs.extend(print_log_lines)
            print(f"[SANDBOX] Print logs: {print_log_lines}")

        # Lấy hàm execute từ scope và gọi nó
        execute_func = execution_scope.get('execute')
        print(f"[SANDBOX] Execute function: {execute_func}")
        
        if not callable(execute_func):
            error_msg = "Không tìm thấy hàm 'execute' trong code của node."
            print(f"[SANDBOX] ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Kiểm tra signature của hàm để gọi đúng tham số
        import inspect
        sig = inspect.signature(execute_func)
        params = list(sig.parameters.keys())
        print(f"[SANDBOX] Execute function parameters: {params}")
        
        if isinstance(input_data, dict) and len(sig.parameters) > 1:
            print(f"[SANDBOX] Multi-parameter function detected")
            # Nếu có nhiều tham số, thử gọi bằng keyword arguments
            try:
                # Lọc chỉ lấy các tham số có trong signature
                valid_params = {k: v for k, v in input_data.items() if k in sig.parameters}
                print(f"[SANDBOX] Valid parameters for function: {valid_params}")
                result = execute_func(**valid_params)
                print(f"[SANDBOX] Function called successfully with kwargs")
            except TypeError as e:
                print(f"[SANDBOX] Kwargs call failed: {e}, trying single parameter")
                # Nếu không thành công, thử gọi theo cách cũ
                result = execute_func(input_data)
                print(f"[SANDBOX] Function called successfully with single param")
        else:
            print(f"[SANDBOX] Single parameter function")
            # Special handling for functions expecting config parameter
            if len(params) == 1 and params[0] == 'config' and 'config' in input_data:
                print(f"[SANDBOX] Passing config parameter directly")
                result = execute_func(input_data['config'])
            else:
                # Gọi theo cách truyền thống
                result = execute_func(input_data)
            print(f"[SANDBOX] Function called successfully")
        
        print(f"[SANDBOX] Function result: {result}")
        
        return {
            "result": result,
            "logs": logs
        }

    except Exception as e:
        error_msg = f"LỖI THỰC THI NODE: {e}"
        print(f"[SANDBOX] {error_msg}")
        console_log(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "result": {"error": str(e)},
            "logs": logs
        }

# --- Sẵn sàng cho tương lai ---
# def execute_javascript_node(code: str, input_data: dict) -> dict:
#     """
#     Sử dụng subprocess để gọi Node.js thực thi code JavaScript.
#     """
#     # import subprocess
#     # ... logic to run node.js script ...
#     pass