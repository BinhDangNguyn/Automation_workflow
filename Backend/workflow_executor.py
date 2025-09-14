# workflow_executor.py
import collections
from node_library import NODE_LIBRARY
from node_sandbox import execute_python_node

class WorkflowExecutor:
    def __init__(self, workflow_data: dict):
        print("[EXECUTOR] Initializing WorkflowExecutor")
        print(f"[EXECUTOR] Input workflow_data keys: {list(workflow_data.keys())}")
        
        self.nodes = {node['id']: node for node in workflow_data['nodes'].values()}
        print(f"[EXECUTOR] Loaded {len(self.nodes)} nodes: {list(self.nodes.keys())}")
        
        self.adj = collections.defaultdict(list)
        self.in_degree = collections.defaultdict(int)
        
        print("[EXECUTOR] Building dependency graph")
        self._build_graph()

    def _build_graph(self):
        """Xây dựng đồ thị phụ thuộc từ dữ liệu workflow."""
        print("[EXECUTOR] Building graph connections:")
        for node_id, node in self.nodes.items():
            print(f"[EXECUTOR]   Node {node_id} outputs: {node.get('outputs', {})}")
            for output in node.get('outputs', {}).values():
                for connection in output.get('connections', []):
                    target_node_id = connection['node']
                    connection_label = connection.get('connection_label', 'unknown')
                    print(f"[EXECUTOR]     Connection: {node_id} -> {target_node_id} ({connection_label})")
                    if target_node_id in self.nodes:
                        self.adj[node_id].append(target_node_id)
                        self.in_degree[target_node_id] += 1
                    else:
                        print(f"[EXECUTOR]     Warning: Target node {target_node_id} not found")
        
        print(f"[EXECUTOR] Final graph adjacency: {dict(self.adj)}")
        print(f"[EXECUTOR] Final in-degrees: {dict(self.in_degree)}")

    def execute(self):
        """Thực thi workflow bằng thuật toán sắp xếp topo với hỗ trợ nhiều connection points."""
        print("[EXECUTOR] Starting workflow execution")
        
        starting_nodes = [node_id for node_id in self.nodes if self.in_degree[node_id] == 0]
        print(f"[EXECUTOR] Starting nodes (in_degree=0): {starting_nodes}")
        
        queue = collections.deque(starting_nodes)
        
        node_results = {}
        all_logs = []
        execution_order = []
        
        all_logs.append("[EXECUTOR] Starting workflow execution")
        all_logs.append(f"[EXECUTOR] Starting nodes: {starting_nodes}")

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)
            node = self.nodes[node_id]
            
            print(f"[EXECUTOR] Processing node {node_id} (type: {node.get('name')})")
            all_logs.append(f"[EXECUTOR] Processing node {node_id} (type: {node.get('name')})")
            
            # 1. Thu thập dữ liệu đầu vào từ các node cha theo connection labels
            input_data = {}
            connection_inputs = {}  # Để lưu trữ theo label của connection
            
            print(f"[EXECUTOR] Collecting inputs for node {node_id}")
            all_logs.append(f"[EXECUTOR] Collecting inputs for node {node_id}")
            
            # Kiểm tra nếu là chat trigger node
            if node.get('data', {}).get('chat_message'):
                input_data['chat_message'] = node['data']['chat_message']
                input_data['message'] = node['data']['chat_message']  # Also set as message
            
            # Include node configuration in input data
            node_config = node.get('data', {}).get('config', {})
            if node_config:
                input_data.update({'node_config': node_config})
            
            for parent_id, parent_node in self.nodes.items():
                for output in parent_node.get('outputs', {}).values():
                    for conn in output.get('connections', []):
                        if conn['node'] == node_id and parent_id in node_results:
                            parent_result = node_results.get(parent_id, {})
                            
                            # Nếu có connection label, lưu theo label
                            connection_label = conn.get('connection_label', 'input')  # Mặc định là 'input'
                            if not connection_label or connection_label == 'input_1':
                                connection_label = 'input'  # Normalize input_1 to input
                            if connection_label not in connection_inputs:
                                connection_inputs[connection_label] = parent_result
                            else:
                                # Nếu đã có, merge data
                                if isinstance(connection_inputs[connection_label], dict) and isinstance(parent_result, dict):
                                    connection_inputs[connection_label].update(parent_result)
                            
                            # Cũng giữ cách cũ để tương thích ngược
                            input_data.update(parent_result)

            # 2. Lấy code của node từ thư viện
            node_type = node.get('name')
            node_definition = NODE_LIBRARY.get(node_type)
            
            if not node_definition:
                all_logs.append(f"Cảnh báo: Không tìm thấy định nghĩa cho node type '{node_type}'. Bỏ qua.")
                continue

            all_logs.append(f"--- Bắt đầu thực thi Node ID: {node_id} (Type: {node_type}, NodeType: {node_definition.get('node_type', 'unknown')}) ---")

            # 3. Chuẩn bị input data dựa trên node type
            all_logs.append(f"Connection inputs: {connection_inputs}")
            all_logs.append(f"Node config: {node_config}")
            
            if node_definition.get('node_type') == 'agent':
                # AI Agent node cần multiple inputs
                execution_input = {
                    'input_data': connection_inputs.get('input', input_data),
                    'model': connection_inputs.get('model'),
                    'memory': connection_inputs.get('memory'),
                    'tools': connection_inputs.get('tools'),
                    'config': node_config
                }
            elif node_definition.get('node_type') == 'system' and node_type in ['and-gate', 'or-gate']:
                # Logic gates cần 2 inputs
                execution_input = {
                    'input1': connection_inputs.get('input1'),
                    'input2': connection_inputs.get('input2'),
                    'config': node_config
                }
            elif node_definition.get('node_type') == 'model':
                # Model nodes use their configuration
                execution_input = {
                    'config': node_config
                }
            else:
                # Các node khác dùng input_data thông thường
                execution_input = input_data or connection_inputs.get('input', {})
                if isinstance(execution_input, dict):
                    execution_input['config'] = node_config
                else:
                    execution_input = {'data': execution_input, 'config': node_config}

            # 4. Thực thi code trong sandbox
            all_logs.append(f"Execution input for {node_type}: {execution_input}")
            
            sandbox_result = execute_python_node(
                code=node_definition['code'],
                input_data=execution_input
            )
            
            node_results[node_id] = sandbox_result['result']
            all_logs.extend(sandbox_result['logs'])
            all_logs.append(f"--- Kết thúc Node ID: {node_id} ---")

            # 5. Thêm các node con vào hàng đợi
            for neighbor_id in self.adj[node_id]:
                self.in_degree[neighbor_id] -= 1
                if self.in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # Lấy kết quả của node cuối cùng làm kết quả chung
        final_result = node_results.get(execution_order[-1], {}) if execution_order else {}

        return {
            "result": final_result,
            "logs": "\\n".join(all_logs),
            "execution_order": execution_order,
            "all_results": node_results
        }