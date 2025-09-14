# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from workflow_executor import WorkflowExecutor
from fastapi.middleware.cors import CORSMiddleware
import json
import requests

# --- Pydantic Models for Data Validation ---
class Connection(BaseModel):
    node: str
    output: str

class Output(BaseModel):
    connections: List[Connection]

class Node(BaseModel):
    id: str
    name: str
    data: Dict[str, Any]
    pos_x: float
    pos_y: float
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Output] = Field(default_factory=dict)

class HomeModule(BaseModel):
    nodes: Dict[str, Node]

class DrawflowData(BaseModel):
    home: HomeModule

class WorkflowPayload(BaseModel):
    drawflow: DrawflowData

# --- FastAPI App ---
app = FastAPI(title="Workflow Automation Backend")

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, hãy thay bằng domain của frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AutoFlow Studio Backend!"}

@app.get("/ollama/models")
async def get_ollama_models():
    """Fetch available models from Ollama"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=503, detail="Ollama service not available")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")

@app.post("/run")
async def run_workflow(payload: dict):
    """
    Endpoint chính để nhận và thực thi một workflow.
    """
    try:
        print("=" * 60)
        print("[MAIN] Starting workflow execution")
        print(f"[MAIN] Received payload keys: {list(payload.keys())}")
        print(f"[MAIN] Full payload: {payload}")
        
        # Handle both direct payload and nested drawflow structure
        if 'drawflow' in payload:
            workflow_data = payload['drawflow']['home']
            print("[MAIN] Using nested drawflow structure")
        else:
            workflow_data = payload
            print("[MAIN] Using direct payload structure")
            
        print(f"[MAIN] Workflow data keys: {list(workflow_data.keys())}")
        print(f"[MAIN] Number of nodes: {len(workflow_data.get('nodes', {}))}")
        
        for node_id, node_data in workflow_data.get('nodes', {}).items():
            print(f"[MAIN] Node {node_id}: type={node_data.get('name')}, config={node_data.get('data', {}).get('config', {})}")
        
        if not workflow_data.get('nodes'):
            print("[MAIN] No nodes found in workflow")
            return {"result": {}, "logs": "Workflow is empty."}
            
        print("[MAIN] Creating WorkflowExecutor")
        executor = WorkflowExecutor(workflow_data)
        
        print("[MAIN] Executing workflow")
        result = executor.execute()
        
        print(f"[MAIN] Workflow execution completed")
        print(f"[MAIN] Result keys: {list(result.keys())}")
        print("=" * 60)
        
        return result
    except Exception as e:
        # Ghi lại lỗi chi tiết ở server
        print("=" * 60)
        print(f"[MAIN] ERROR executing workflow: {e}") 
        import traceback
        traceback.print_exc()
        print("=" * 60)
        # Trả về lỗi chung cho client
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    # Chạy server trên cổng 5000 để khớp với frontend
    uvicorn.run(app, host="127.0.0.1", port=5000)