# node_library.py
import json
import time
from llm import LLMChatBot, ToolManager, MemoryBuffer
from memory_manager import memory_manager

# Console logging helper
def console_log(message):
    """Simple console logging for node execution"""
    print(f"[NODE] {message}")

# Enhanced helper functions
def call_llm_agent(prompt: str, model_config=None, memory_config=None, tools_config=None) -> dict:
    """Enhanced LLM agent caller with configurable components"""
    try:
        # Use model config if provided
        if model_config:
            model_name = model_config.get('model', '')
            if not model_name:
                return {"error": "No model specified in configuration"}
                
            api_url = model_config.get('api_url') or model_config.get('url', 'http://localhost:11434') + '/api/chat'
            provider = model_config.get('provider', 'ollama')
            capabilities = model_config.get('capabilities', {})
        else:
            return {"error": "No model configuration provided"}
        
        console_log(f"Using LLM: {model_name} via {api_url}")
        console_log(f"Model capabilities: {capabilities}")
        
        # Setup tools - only if model supports tool calling
        tool_manager = None
        if tools_config and 'tools' in tools_config and capabilities.get('tool_calling', False):
            tool_manager = ToolManager("tools")
            enabled_tools = [t['name'] for t in tools_config['tools'] if t.get('enabled', True)]
            console_log(f"Enabled tools: {enabled_tools}")
        elif tools_config and 'tools' in tools_config:
            console_log("Warning: Model doesn't support tool calling, skipping tools")
        else:
            console_log("No tools configured or no tool calling support")
        
        # Setup memory
        if memory_config:
            memory_id = f"agent_{hash(prompt)}_{memory_config.get('type', 'buffer')}"
            if memory_manager.get_memory(memory_id) is None:
                memory_instance = memory_manager.create_memory(memory_id, memory_config)
            else:
                memory_instance = memory_manager.get_memory(memory_id)
            
            # Store current prompt
            memory_instance.store(f"prompt_{int(time.time())}", prompt)
            console_log(f"Using memory: {memory_config.get('type', 'buffer')}")
            
        # Use minimal system prompt to get raw responses
        system_prompt = ""
        memory = MemoryBuffer(system_prompt)
        
        # Create LLMChatBot with optional tool_manager
        bot = LLMChatBot(
            model_name=model_name,
            api_url=api_url,
            memory_buffer=memory,
            tool_manager=tool_manager  # Can be None for non-tool models
        )
        
        console_log(f"Calling LLM with prompt: {prompt[:100]}...")
        result_content = bot.ask_once(prompt)
        console_log(f"LLM raw output length: {len(str(result_content))}")
        console_log(f"LLM raw output: {result_content}")
        
        return {
            "result": result_content, 
            "model_used": model_name,
            "capabilities_used": capabilities,
            "provider": provider
        }
    except Exception as e:
        console_log(f"Error in call_llm_agent: {e}")
        return {"error": str(e)}

def setup_memory_system(memory_config: dict) -> dict:
    """Setup memory system based on configuration"""
    try:
        memory_id = f"memory_{memory_config.get('type', 'buffer')}_{hash(str(memory_config))}"
        memory_instance = memory_manager.create_memory(memory_id, memory_config)
        return {"memory_id": memory_id, "type": memory_config.get('type'), "status": "created"}
    except Exception as e:
        return {"error": str(e)}

def get_available_tools(tools_config: dict) -> list:
    """Get list of available tools based on configuration"""
    if 'tools' in tools_config:
        return [t for t in tools_config['tools'] if t.get('enabled', True)]
    return []

# Định nghĩa code cho từng node
NODE_LIBRARY = {
    # Trigger nodes
    'start': {
        "language": "python",
        "node_type": "trigger",
        "code": """
def execute(input_data=None):
    # Node bắt đầu thường chỉ trả về một trigger message
    return {"status": "started", "initial_data": input_data or {}, "trigger_time": "now"}
"""
    },
    'webhook': {
        "language": "python",
        "node_type": "trigger",
        "code": """
def execute(input_data=None):
    method = input_data.get('method', 'POST') if input_data else 'POST'
    path = input_data.get('path', '/webhook') if input_data else '/webhook'
    console_log(f"Webhook triggered: {method} {path}")
    return {"method": method, "path": path, "triggered": True}
"""
    },
    'schedule': {
        "language": "python",
        "node_type": "trigger",
        "code": """
def execute(input_data=None):
    interval = input_data.get('interval', 'daily') if input_data else 'daily'
    console_log(f"Schedule trigger activated: {interval}")
    return {"interval": interval, "triggered_at": "now"}
"""
    },
    'chat': {
        "language": "python",
        "node_type": "trigger",
        "code": """
def execute(input_data=None):
    # Chat trigger - handles incoming messages
    if input_data and 'chat_message' in input_data:
        message = input_data['chat_message']
    else:
        message = input_data.get('message', '') if input_data else ''
    
    console_log(f"Chat trigger received message: {message}")
    return {
        "message": message,
        "timestamp": "now",
        "user": "user",
        "trigger_type": "chat"
    }
"""
    },
    
    # AI Agent nodes
    'ai-agent': {
        "language": "python",
        "node_type": "agent",
        "code": """
def execute(input_data, model=None, memory=None, tools=None, config=None):
    # Advanced AI agent - processes raw LLM calls using connected components
    console_log(f"AI Agent execute called")
    console_log(f"Input data: {input_data}")
    console_log(f"Connected model: {model}")
    console_log(f"Connected memory: {memory}")
    console_log(f"Connected tools: {tools}")
    console_log(f"Agent config: {config}")
    
    # Extract input message
    if isinstance(input_data, dict):
        if 'message' in input_data:
            prompt = input_data['message']
        elif 'chat_message' in input_data:
            prompt = input_data['chat_message']
        else:
            prompt = input_data.get('prompt', str(input_data))
    else:
        prompt = str(input_data)
    
    console_log(f"Processing message: {prompt}")
    
    # Check if model is connected and valid
    if not model or model.get('error'):
        error_msg = model.get('error', 'No model connected') if model else 'No model connected'
        console_log(f"Error: {error_msg}")
        return {
            "error": error_msg,
            "raw_response": "",
            "model_info": model,
            "processed": False
        }
    
    # Get system prompt from agent config
    system_prompt = ""
    if config and 'system_prompt' in config and config['system_prompt']:
        system_prompt = config['system_prompt']
        console_log(f"Using system prompt: {system_prompt[:50]}{'...'}")
    
    # Prepare full prompt
    if system_prompt:
        full_prompt = system_prompt + "\\n\\nUser: " + prompt
    else:
        full_prompt = prompt
    
    console_log(f"Full prompt: {full_prompt}")
    
    # Call LLM with connected model configuration
    console_log(f"Calling LLM with model: {model.get('model')} from {model.get('provider')}")
    llm_response = call_llm_agent(full_prompt, model, memory, tools)
    
    console_log(f"Raw LLM response: {llm_response}")
    
    # Return raw response - let other nodes handle post-processing
    return {
        "raw_response": llm_response.get('result', '') if isinstance(llm_response, dict) else str(llm_response),
        "model_info": {
            "provider": model.get('provider', 'unknown'),
            "model": model.get('model', 'unknown'),
            "capabilities": model.get('capabilities', {})
        },
        "memory_used": bool(memory),
        "tools_used": bool(tools),
        "system_prompt_used": bool(system_prompt),
        "processed": True,
        "full_llm_response": llm_response  # For debugging
    }
"""
    },
    'llm-agent': {
        "language": "python",
        "node_type": "processing",
        "code": """
def execute(input_data):
    prompt = input_data.get('prompt', 'Xin chào, bạn là ai?') if isinstance(input_data, dict) else str(input_data)
    console_log(f"Đang gửi prompt tới LLM Agent: {prompt}")
    llm_response = call_llm_agent(prompt)
    return {"response": llm_response}
"""
    },
    
    # System Logic nodes
    'if-else': {
        "language": "python",
        "node_type": "system",
        "code": """
def execute(input_data):
    condition = input_data.get('condition', True) if isinstance(input_data, dict) else bool(input_data)
    console_log(f"If-Else evaluating condition: {condition}")
    
    if condition:
        return {"result": True, "branch": "true", "data": input_data}
    else:
        return {"result": False, "branch": "false", "data": input_data}
"""
    },
    'and-gate': {
        "language": "python",
        "node_type": "system",
        "code": """
def execute(input1=None, input2=None):
    val1 = bool(input1) if input1 is not None else False
    val2 = bool(input2) if input2 is not None else False
    result = val1 and val2
    console_log(f"AND Gate: {val1} AND {val2} = {result}")
    return {"result": result, "input1": val1, "input2": val2}
"""
    },
    'or-gate': {
        "language": "python",
        "node_type": "system",
        "code": """
def execute(input1=None, input2=None):
    val1 = bool(input1) if input1 is not None else False
    val2 = bool(input2) if input2 is not None else False
    result = val1 or val2
    console_log(f"OR Gate: {val1} OR {val2} = {result}")
    return {"result": result, "input1": val1, "input2": val2}
"""
    },
    'json-passthrough': {
        "language": "python",
        "node_type": "system",
        "code": """
def execute(input_data):
    # System logic node that takes JSON input and returns it unchanged
    console_log("--- JSON PASSTHROUGH NODE ---")
    console_log(f"Input type: {type(input_data)}")
    console_log(f"Input data: {input_data}")
    
    # Return exactly what was received without any modifications
    return input_data
"""
    },
    
    # Data Processing nodes
    'web-search': {
        "language": "python",
        "node_type": "processing",
        "code": """
from tools import web_search

def execute(input_data):
    query = input_data.get('query', 'tin tức mới nhất') if isinstance(input_data, dict) else str(input_data)
    console_log(f"Đang thực hiện web search với query: {query}")
    results = web_search(query=query, num_results=3)
    return {"search_results": results, "query": query}
"""
    },
    'data-transform': {
        "language": "python",
        "node_type": "processing",
        "code": """
def execute(input_data):
    console_log(f"Transforming data: {input_data}")
    # Example transformation - you can customize this
    if isinstance(input_data, dict):
        transformed = {k.upper(): v for k, v in input_data.items()}
    else:
        transformed = str(input_data).upper()
    
    return {"transformed": transformed, "original": input_data}
"""
    },
    'custom-code': {
        "language": "python",
        "node_type": "processing",
        "code": """
def execute(input_data):
    # Custom code node for flexible output parsing
    console_log("--- CUSTOM CODE NODE ---")
    console_log(f"Input: {input_data}")
    
    # ============================================
    # CUSTOMIZE YOUR PROCESSING LOGIC BELOW
    # ============================================
    
    # Example 1: Extract thinking tags from text
    if isinstance(input_data, dict) and 'raw_response' in input_data:
        raw_text = input_data['raw_response']
        if '<think>' in raw_text and '</think>' in raw_text:
            import re
            thinking_pattern = r'<think>(.*?)</think>'
            thinking_match = re.search(thinking_pattern, raw_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                final_response = re.sub(thinking_pattern, '', raw_text, flags=re.DOTALL).strip()
                return {
                    "thinking": thinking_content,
                    "response": final_response,
                    "has_thinking": True
                }
        return {"response": raw_text, "has_thinking": False}
    
    # Example 2: Process JSON and extract specific fields
    if isinstance(input_data, dict):
        # Extract specific fields you need
        result = {}
        for key, value in input_data.items():
            if isinstance(value, str):
                result[f"processed_{key}"] = value.strip().upper()
            else:
                result[f"processed_{key}"] = value
        return result
    
    # Example 3: Convert any input to structured format
    if isinstance(input_data, str):
        return {
            "original_text": input_data,
            "word_count": len(input_data.split()),
            "char_count": len(input_data),
            "processed_at": "Custom Code Node"
        }
    
    # Default: return input as-is with metadata
    return {
        "data": input_data,
        "type": str(type(input_data)),
        "processed": True,
        "message": "Customize this node for your specific parsing needs"
    }
    
    # ============================================
    # END CUSTOMIZATION AREA
    # ============================================
"""
    },
    
    # Output nodes
    'chat-output': {
        "language": "python",
        "node_type": "output",
        "code": """
def execute(input_data):
    # Simply output raw values without any processing
    console_log("--- CHAT OUTPUT NODE ---")
    
    if isinstance(input_data, dict):
        # Extract response from AI agent
        raw_response = input_data.get('raw_response', '')
        if not raw_response:
            raw_response = input_data.get('result', str(input_data))
    else:
        raw_response = str(input_data)
    
    console_log(f"Raw output: {raw_response}")
    
    # Return raw value without any processing
    return {
        "chat_response": raw_response,
        "raw_value": raw_response,
        "processed": False
    }
"""
    },
    'log-message': {
        "language": "python",
        "node_type": "output",
        "code": """
def execute(input_data):
    message = json.dumps(input_data, indent=2, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)
    console_log("--- LOG MESSAGE NODE ---")
    console_log(message)
    console_log("----------------------")
    return {"logged": True, "message": message}
"""
    },
    'send-email': {
        "language": "python",
        "node_type": "output",
        "code": """
def execute(input_data):
    # Mock email sending - replace with actual email logic
    to = input_data.get('to', 'user@example.com') if isinstance(input_data, dict) else 'user@example.com'
    subject = input_data.get('subject', 'Workflow Notification') if isinstance(input_data, dict) else 'Workflow Notification'
    body = input_data.get('body', str(input_data)) if isinstance(input_data, dict) else str(input_data)
    
    console_log(f"Sending email to: {to}")
    console_log(f"Subject: {subject}")
    console_log(f"Body: {body}")
    
    return {"sent": True, "to": to, "subject": subject}
"""
    },
    
    # Model Nodes
    'chatgpt': {
        "language": "python",
        "node_type": "model",
        "code": """
def execute(config=None):
    if config:
        model = config.get('model', 'gpt-4')
        api_key = config.get('api_key', 'your-openai-api-key')
    else:
        model = 'gpt-4'
        api_key = 'your-openai-api-key'
    
    console_log(f"ChatGPT model configured: {model}")
    
    return {
        "provider": "openai",
        "model": model,
        "api_key": api_key,
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 4000,
        "temperature": 0.7
    }
"""
    },
    'gemini': {
        "language": "python",
        "node_type": "model",
        "code": """
def execute():
    return {
        "provider": "google",
        "model": "gemini-pro",
        "api_key": "your-google-api-key",
        "max_tokens": 4000,
        "temperature": 0.7
    }
"""
    },
    'ollama': {
        "language": "python",
        "node_type": "model",
        "code": """
def execute(config=None):
    if not config or not config.get('model'):
        console_log("Warning: No model selected in Ollama node configuration")
        return {
            "provider": "ollama",
            "model": "",
            "url": "http://localhost:11434",
            "error": "No model selected"
        }
    
    model = config.get('model')
    url = config.get('url', 'http://localhost:11434')
    
    # Detect model capabilities based on model name
    def detect_capabilities(model_name):
        name = model_name.lower()
        capabilities = {
            'thinking': False,
            'tool_calling': False,
            'vision': False,
            'coding': False,
            'reasoning': False,
            'multimodal': False
        }
        
        # Thinking models
        if 'qwq' in name or 'deepseek-r1' in name or 'o1' in name or 'think' in name:
            capabilities['thinking'] = True
            capabilities['reasoning'] = True
        
        # Tool calling models (be more specific)
        if any(x in name for x in ['llama3.1', 'llama3.2', 'mistral', 'qwen2.5', 'qwen3', 'deepseek-v3', 'codestral']):
            capabilities['tool_calling'] = True
        # Explicitly disable for models known not to support tool calling
        elif any(x in name for x in ['deepseek-r1', 'gemma', 'llama3:8b', 'llama3:7b']):
            capabilities['tool_calling'] = False
            
        # Vision models
        if any(x in name for x in ['vision', 'llava', 'minicpm', 'qwen-vl', 'phi-3-vision']):
            capabilities['vision'] = True
            capabilities['multimodal'] = True
            
        # Coding models
        if any(x in name for x in ['code', 'deepseek-coder', 'codestral', 'starcoder', 'granite-code']):
            capabilities['coding'] = True
            
        return capabilities
    
    capabilities = detect_capabilities(model)
    console_log(f"Ollama model configured: {model} at {url}")
    console_log(f"Model capabilities: {capabilities}")
    
    return {
        "provider": "ollama",
        "model": model,
        "url": url,
        "api_url": f"{url}/api/chat",
        "capabilities": capabilities,
        "max_tokens": 4000,
        "temperature": 0.7
    }
"""
    },
    
    # Memory Nodes
    'buffer-memory': {
        "language": "python",
        "node_type": "memory",
        "code": """
def execute():
    return {
        "type": "buffer",
        "max_tokens": 4000,
        "system_prompt": "",
        "history": []
    }
"""
    },
    'short-term-memory': {
        "language": "python",
        "node_type": "memory",
        "code": """
def execute():
    return {
        "type": "short_term",
        "ttl": 3600,  # 1 hour
        "max_items": 100,
        "storage": "memory",
        "cleanup_policy": "auto"
    }
"""
    },
    'long-term-memory': {
        "language": "python",
        "node_type": "memory",
        "code": """
def execute():
    return {
        "type": "long_term",
        "storage": "file",
        "path": "./memory.json",
        "compression": True,
        "encryption": False
    }
"""
    },
    'vector-memory': {
        "language": "python",
        "node_type": "memory",
        "code": """
def execute():
    return {
        "type": "vector",
        "embedding_model": "sentence-transformers",
        "index_type": "faiss",
        "dimensions": 384,
        "similarity_threshold": 0.7
    }
"""
    },
    
    # Tool Nodes
    'web-tools': {
        "language": "python",
        "node_type": "tools",
        "code": """
def execute():
    return {
        "tools": [
            {"name": "web_search", "enabled": True, "description": "Search the web"},
            {"name": "web_scrape", "enabled": True, "description": "Scrape web content"},
            {"name": "url_fetch", "enabled": False, "description": "Fetch URL content"}
        ]
    }
"""
    },
    'file-tools': {
        "language": "python",
        "node_type": "tools",
        "code": """
def execute():
    return {
        "tools": [
            {"name": "read_file", "enabled": True, "description": "Read file content"},
            {"name": "write_file", "enabled": True, "description": "Write to file"},
            {"name": "list_directory", "enabled": False, "description": "List directory contents"}
        ]
    }
"""
    },
    'code-tools': {
        "language": "python",
        "node_type": "tools",
        "code": """
def execute():
    return {
        "tools": [
            {"name": "execute_python", "enabled": True, "description": "Execute Python code"},
            {"name": "analyze_code", "enabled": False, "description": "Analyze code structure"},
            {"name": "format_code", "enabled": False, "description": "Format code"}
        ]
    }
"""
    }
}