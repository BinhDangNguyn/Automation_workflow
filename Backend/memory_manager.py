# memory_manager.py - Enhanced Memory Management System

import json
import time
import sqlite3
import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import hashlib

class BaseMemory(ABC):
    """Base class for all memory types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.type = config.get('type', 'unknown')
    
    @abstractmethod
    def store(self, key: str, data: Any) -> bool:
        """Store data with a key"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by query"""
        pass
    
    @abstractmethod
    def cleanup(self) -> int:
        """Clean up old memories, return number of items removed"""
        pass

class BufferMemory(BaseMemory):
    """Simple conversation buffer memory"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_tokens = config.get('max_tokens', 4000)
        self.system_prompt = config.get('system_prompt', '')
        self.history = []
    
    def store(self, key: str, data: Any) -> bool:
        entry = {
            'key': key,
            'data': data,
            'timestamp': time.time(),
            'tokens': len(str(data))  # Simple token estimation
        }
        self.history.append(entry)
        self._enforce_token_limit()
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        for entry in reversed(self.history):
            if entry['key'] == key:
                return entry['data']
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()
        
        for entry in reversed(self.history):
            if query_lower in str(entry['data']).lower():
                results.append({
                    'key': entry['key'],
                    'data': entry['data'],
                    'timestamp': entry['timestamp'],
                    'relevance': 1.0
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def cleanup(self) -> int:
        old_count = len(self.history)
        self.history = []
        return old_count
    
    def _enforce_token_limit(self):
        total_tokens = sum(entry['tokens'] for entry in self.history)
        while total_tokens > self.max_tokens and self.history:
            removed = self.history.pop(0)
            total_tokens -= removed['tokens']

class ShortTermMemory(BaseMemory):
    """Short-term memory with TTL"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ttl = config.get('ttl', 3600)  # 1 hour default
        self.max_items = config.get('max_items', 100)
        self.storage = {}
    
    def store(self, key: str, data: Any) -> bool:
        self.storage[key] = {
            'data': data,
            'timestamp': time.time(),
            'expires_at': time.time() + self.ttl
        }
        self._enforce_item_limit()
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        if key in self.storage:
            entry = self.storage[key]
            if time.time() < entry['expires_at']:
                return entry['data']
            else:
                del self.storage[key]  # Expired
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._cleanup_expired()
        results = []
        query_lower = query.lower()
        
        for key, entry in self.storage.items():
            if query_lower in str(entry['data']).lower():
                results.append({
                    'key': key,
                    'data': entry['data'],
                    'timestamp': entry['timestamp'],
                    'relevance': 1.0
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def cleanup(self) -> int:
        return self._cleanup_expired()
    
    def _cleanup_expired(self) -> int:
        current_time = time.time()
        expired_keys = [k for k, v in self.storage.items() if current_time >= v['expires_at']]
        for key in expired_keys:
            del self.storage[key]
        return len(expired_keys)
    
    def _enforce_item_limit(self):
        if len(self.storage) > self.max_items:
            # Remove oldest items
            sorted_items = sorted(self.storage.items(), key=lambda x: x[1]['timestamp'])
            items_to_remove = len(self.storage) - self.max_items
            for i in range(items_to_remove):
                del self.storage[sorted_items[i][0]]

class LongTermMemory(BaseMemory):
    """Persistent long-term memory"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_type = config.get('storage', 'file')
        self.path = config.get('path', './memory.json')
        
        if self.storage_type == 'sqlite':
            self._init_sqlite()
        elif self.storage_type == 'file':
            self._ensure_file_exists()
    
    def _init_sqlite(self):
        self.db_path = self.path.replace('.json', '.db')
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                data TEXT,
                timestamp REAL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _ensure_file_exists(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump({}, f)
    
    def store(self, key: str, data: Any) -> bool:
        if self.storage_type == 'sqlite':
            return self._store_sqlite(key, data)
        else:
            return self._store_file(key, data)
    
    def _store_sqlite(self, key: str, data: Any) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                'INSERT OR REPLACE INTO memories (key, data, timestamp, metadata) VALUES (?, ?, ?, ?)',
                (key, json.dumps(data), time.time(), json.dumps({}))
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing to SQLite: {e}")
            return False
    
    def _store_file(self, key: str, data: Any) -> bool:
        try:
            with open(self.path, 'r') as f:
                memories = json.load(f)
            
            memories[key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            with open(self.path, 'w') as f:
                json.dump(memories, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error storing to file: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        if self.storage_type == 'sqlite':
            return self._retrieve_sqlite(key)
        else:
            return self._retrieve_file(key)
    
    def _retrieve_sqlite(self, key: str) -> Optional[Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT data FROM memories WHERE key = ?', (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            print(f"Error retrieving from SQLite: {e}")
            return None
    
    def _retrieve_file(self, key: str) -> Optional[Any]:
        try:
            with open(self.path, 'r') as f:
                memories = json.load(f)
            
            if key in memories:
                return memories[key]['data']
            return None
        except Exception as e:
            print(f"Error retrieving from file: {e}")
            return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.storage_type == 'sqlite':
            return self._search_sqlite(query, limit)
        else:
            return self._search_file(query, limit)
    
    def _search_sqlite(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                'SELECT key, data, timestamp FROM memories WHERE data LIKE ? ORDER BY timestamp DESC LIMIT ?',
                (f'%{query}%', limit)
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    'key': row[0],
                    'data': json.loads(row[1]),
                    'timestamp': row[2],
                    'relevance': 1.0
                })
            conn.close()
            return results
        except Exception as e:
            print(f"Error searching SQLite: {e}")
            return []
    
    def _search_file(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            with open(self.path, 'r') as f:
                memories = json.load(f)
            
            results = []
            query_lower = query.lower()
            
            for key, entry in memories.items():
                if query_lower in str(entry['data']).lower():
                    results.append({
                        'key': key,
                        'data': entry['data'],
                        'timestamp': entry['timestamp'],
                        'relevance': 1.0
                    })
                    if len(results) >= limit:
                        break
            
            return sorted(results, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            print(f"Error searching file: {e}")
            return []
    
    def cleanup(self) -> int:
        # For long-term memory, cleanup could mean removing duplicates or very old entries
        # This is a simplified implementation
        return 0

class VectorMemory(BaseMemory):
    """Semantic vector search memory (simplified implementation)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_model = config.get('embedding_model', 'sentence-transformers')
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.vectors = {}  # In a real implementation, this would use FAISS or similar
    
    def _get_embedding(self, text: str) -> List[float]:
        # Simplified embedding - in real implementation, use actual embedding model
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert hash to pseudo-embedding
        embedding = []
        for i in range(0, len(hash_obj.hexdigest()), 2):
            embedding.append(int(hash_obj.hexdigest()[i:i+2], 16) / 255.0)
        
        # Pad or truncate to fixed size
        target_size = 384
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        # Simplified cosine similarity
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def store(self, key: str, data: Any) -> bool:
        text = str(data)
        embedding = self._get_embedding(text)
        
        self.vectors[key] = {
            'data': data,
            'embedding': embedding,
            'timestamp': time.time()
        }
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        if key in self.vectors:
            return self.vectors[key]['data']
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        results = []
        
        for key, entry in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, entry['embedding'])
            if similarity >= self.similarity_threshold:
                results.append({
                    'key': key,
                    'data': entry['data'],
                    'timestamp': entry['timestamp'],
                    'relevance': similarity
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
    
    def cleanup(self) -> int:
        # Remove low-relevance or very old vectors
        return 0

class MemoryManager:
    """Central memory management system"""
    
    def __init__(self):
        self.memories: Dict[str, BaseMemory] = {}
    
    def create_memory(self, memory_id: str, config: Dict[str, Any]) -> BaseMemory:
        """Create a new memory instance"""
        memory_type = config.get('type', 'buffer')
        
        if memory_type == 'buffer':
            memory = BufferMemory(config)
        elif memory_type == 'short_term':
            memory = ShortTermMemory(config)
        elif memory_type == 'long_term':
            memory = LongTermMemory(config)
        elif memory_type == 'vector':
            memory = VectorMemory(config)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        self.memories[memory_id] = memory
        return memory
    
    def get_memory(self, memory_id: str) -> Optional[BaseMemory]:
        """Get existing memory instance"""
        return self.memories.get(memory_id)
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup all memories"""
        results = {}
        for memory_id, memory in self.memories.items():
            results[memory_id] = memory.cleanup()
        return results
    
    def get_memory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all memories"""
        stats = {}
        for memory_id, memory in self.memories.items():
            if hasattr(memory, 'history'):
                item_count = len(memory.history)
            elif hasattr(memory, 'storage'):
                item_count = len(memory.storage)
            elif hasattr(memory, 'vectors'):
                item_count = len(memory.vectors)
            else:
                item_count = 0
            
            stats[memory_id] = {
                'type': memory.type,
                'item_count': item_count,
                'config': memory.config
            }
        
        return stats

# Global memory manager instance
memory_manager = MemoryManager()