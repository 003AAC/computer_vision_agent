"""
Experience Store - JSON 经验存储基类
====================================
提供原子读写、分类加载基础能力。
"""
import json
import os
from typing import Dict, Any, List, Optional


class ExperienceStore:
    """JSON 文件存储基类"""

    def __init__(self, path: str, default_data: Optional[Dict[str, Any]] = None):
        self.path = path
        self._default = default_data or {}
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self):
        """加载数据，不存在则用默认值并保存"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                return
            except Exception as e:
                print(f"  [Memory] 加载失败 {self.path}: {e}")
        self._data = json.loads(json.dumps(self._default))
        self.save()

    def save(self):
        """原子保存：先写临时文件再替换"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"  [Memory] 保存失败 {self.path}: {e}")

    def all(self) -> Dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self.save()