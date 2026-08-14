"""
知识库模块 - 为 Agent 提供知识和经验检索
包含：
  - 静态知识：环境信息、操作策略、错误模式（JSON 存储）
  - 动态经验：Agent 执行过程中沉淀的经验（向量检索）
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings


class KnowledgeBase:
    """Agent 知识库（带自减负机制）"""
    
    # 自减负配置
    MAX_EXPERIENCES = 1000
    MAX_AGE_DAYS = 60
    MAX_INACTIVE_DAYS = 30
    PROMOTE_ACCESS_COUNT = 5
    
    CATEGORY_LIMITS = {
        "文件操作": 100, "应用启动": 50, "环境调试": 200,
        "GUI操作": 200, "系统管理": 100, "default": 350
    }
    
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "knowledge_data"
            )
        self.knowledge_dir = Path(knowledge_dir).absolute()
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self.env_file = self.knowledge_dir / "environment.json"
        self.strategy_file = self.knowledge_dir / "strategies.json"
        self.error_file = self.knowledge_dir / "error_patterns.json"
        
        chroma_cache_dir = self.knowledge_dir / "chroma_cache"
        chroma_cache_dir.mkdir(exist_ok=True)
        os.environ["CHROMA_HOME"] = str(chroma_cache_dir)
        
        chroma_db_dir = self.knowledge_dir / "chroma_db"
        chroma_db_dir.mkdir(exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.experience_collection = self.chroma_client.get_or_create_collection(
            name="agent_experiences",
            metadata={"description": "Agent执行经验，用于跨任务复用"}
        )
        
        self.environment = self._load_json(self.env_file, default={})
        self.strategies = self._load_json(self.strategy_file, default={})
        self.error_patterns = self._load_json(self.error_file, default={})
    
    def _load_json(self, file_path: Path, default=None) -> Dict:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default if default is not None else {}
    
    def _save_json(self, file_path: Path, data: Dict):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # ==================== 静态知识管理 ====================
    
    def get_environment_info(self) -> Dict:
        return self.environment
    
    def get_strategy(self, task_type: str) -> Optional[Dict]:
        return self.strategies.get(task_type)
    
    def get_error_solution(self, error_keyword: str) -> Optional[Dict]:
        for pattern, solution in self.error_patterns.items():
            if pattern.lower() in error_keyword.lower():
                return solution
        return None
    
    def update_environment(self, key: str, value):
        self.environment[key] = value
        self._save_json(self.env_file, self.environment)
    
    def update_strategy(self, task_type: str, strategy: Dict):
        self.strategies[task_type] = strategy
        self._save_json(self.strategy_file, self.strategies)
    
    def update_error_pattern(self, pattern: str, solution: Dict):
        self.error_patterns[pattern] = solution
        self._save_json(self.error_file, self.error_patterns)
    
    # ==================== 动态经验管理 ====================
    
    def add_experience(self, experience: Dict):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        category = self._auto_categorize(experience.get('task', ''), experience.get('tools_used', []))
        
        doc_text = f"""
任务：{experience['task']}
结果：{experience['result']}
步数：{experience.get('steps', '未知')}
类别：{category}
主要工具：{', '.join(experience.get('tools_used', []))}
经验：{experience['lesson']}
日期：{experience['date']}
标签：{', '.join(experience.get('tags', []))}
"""
        doc_id = f"{experience['date']}_{hash(experience['task']) % 10000}"
        tools_used = experience.get('tools_used', [])
        tool_counts = experience.get('tool_counts', {})
        exp_type = experience.get('_type', 'experience')
        
        self.experience_collection.add(
            documents=[doc_text],
            metadatas=[{
                "task": experience['task'],
                "result": experience['result'],
                "lesson": experience['lesson'],
                "date": experience['date'],
                "steps": experience.get('steps', 0),
                "tools_used": json.dumps(tools_used, ensure_ascii=False),
                "tool_counts": json.dumps(tool_counts, ensure_ascii=False),
                "tags": json.dumps(experience.get('tags', []), ensure_ascii=False),
                "category": category,
                "_type": exp_type,
                "created_time": now_str,
                "last_accessed": now_str,
                "access_count": 0
            }],
            ids=[doc_id]
        )
        self._auto_prune()
    
    def _auto_categorize(self, task: str, tools_used: list) -> str:
        task_lower = task.lower()
        tools_str = ','.join(tools_used).lower()
        
        if any(kw in task for kw in ['创建文件', '新建', '删除文件', '重命名', '移动', '复制']):
            return '文件操作'
        if any(kw in task for kw in ['打开', '启动', '运行']):
            if any(kw in task for kw in ['记事本', '浏览器', 'WPS', 'VS', '游戏', '微信', 'QQ']):
                return '应用启动'
        if any(kw in task_lower for kw in ['error', '错误', '失败', '报错', 'bug', '调试', '修复']):
            return '环境调试'
        if 'run_command' in tools_str and any(kw in tools_str for kw in ['pip', 'install', 'set', 'env']):
            return '环境调试'
        if 'screenshot' in tools_str and 'run_command' not in tools_str:
            if any(kw in task for kw in ['游戏', '网页', '界面', '点击', '菜单']):
                return 'GUI操作'
        if any(kw in task for kw in ['环境变量', '注册表', '服务', '权限', '设置']):
            return '系统管理'
        if 'run_command' in tools_str:
            return '文件操作'
        if 'click' in tools_str or 'type' in tools_str:
            return 'GUI操作'
        return 'default'
    
    def search_experience(self, query: str, n_results: int = 3) -> List[Dict]:
        if self.experience_collection.count() == 0:
            return []
        
        results = self.experience_collection.query(query_texts=[query], n_results=n_results)
        experiences = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        ids_to_update = []
        
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            old_count = meta.get('access_count', 0)
            new_count = old_count + 1
            meta['last_accessed'] = now_str
            meta['access_count'] = new_count
            ids_to_update.append({
                'id': results['ids'][0][i],
                'document': doc, 'metadata': meta
            })
            experiences.append({
                'document': doc, 'metadata': meta,
                'distance': results['distances'][0][i]
            })
        
        if ids_to_update:
            try:
                self.experience_collection.update(
                    ids=[item['id'] for item in ids_to_update],
                    metadatas=[item['metadata'] for item in ids_to_update],
                    documents=[item['document'] for item in ids_to_update]
                )
            except Exception:
                pass
        return experiences
    
    # ==================== 自减负机制 ====================
    
    def _auto_prune(self):
        total = self.experience_collection.count()
        try:
            all_data = self.experience_collection.get(include=['metadatas'])
            now = datetime.now()
            category_counts = {}
            for meta in all_data['metadatas']:
                cat = meta.get('category', 'default')
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            ids_to_delete = []
            for cat, count in category_counts.items():
                limit = self.CATEGORY_LIMITS.get(cat, self.CATEGORY_LIMITS['default'])
                if count > limit:
                    cat_items = []
                    for i, meta in enumerate(all_data['metadatas']):
                        if meta.get('category', 'default') != cat:
                            continue
                        access_count = meta.get('access_count', 0)
                        if access_count >= self.PROMOTE_ACCESS_COUNT:
                            continue
                        created_str = meta.get('created_time', meta.get('date', ''))
                        try:
                            if ' ' in created_str:
                                created = datetime.strptime(created_str, "%Y-%m-%d %H:%M")
                            else:
                                created = datetime.strptime(created_str, "%Y-%m-%d")
                            age_days = (now - created).days
                        except (ValueError, TypeError):
                            age_days = 0
                        cat_items.append({'id': all_data['ids'][i], 'age_days': age_days, 'access_count': access_count})
                    cat_items.sort(key=lambda x: -x['age_days'])
                    excess = count - limit
                    ids_to_delete.extend([item['id'] for item in cat_items[:excess]])
            
            if total > self.MAX_EXPERIENCES:
                scored_items = []
                for i, meta in enumerate(all_data['metadatas']):
                    doc_id = all_data['ids'][i]
                    if doc_id in ids_to_delete:
                        continue
                    access_count = meta.get('access_count', 0)
                    if access_count >= self.PROMOTE_ACCESS_COUNT:
                        continue
                    created_str = meta.get('created_time', meta.get('date', ''))
                    try:
                        if ' ' in created_str:
                            created = datetime.strptime(created_str, "%Y-%m-%d %H:%M")
                        else:
                            created = datetime.strptime(created_str, "%Y-%m-%d")
                        age_days = (now - created).days
                    except (ValueError, TypeError):
                        age_days = 0
                    last_accessed_str = meta.get('last_accessed', created_str)
                    try:
                        if ' ' in last_accessed_str:
                            last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d %H:%M")
                        else:
                            last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d")
                        inactive_days = (now - last_accessed).days
                    except (ValueError, TypeError):
                        inactive_days = age_days
                    score = 0
                    if age_days > self.MAX_AGE_DAYS:
                        score += 10
                    if inactive_days > self.MAX_INACTIVE_DAYS:
                        score += 10
                    score += age_days
                    score -= access_count * 2
                    scored_items.append({'id': doc_id, 'score': score})
                scored_items.sort(key=lambda x: -x['score'])
                global_excess = (total - len(ids_to_delete)) - self.MAX_EXPERIENCES
                if global_excess > 0:
                    ids_to_delete.extend([item['id'] for item in scored_items[:global_excess]])
            
            if ids_to_delete:
                self.experience_collection.delete(ids=ids_to_delete)
                print(f"✓ 自动清理 {len(ids_to_delete)} 条过期经验，剩余 {total - len(ids_to_delete)} 条")
        except Exception as e:
            print(f"✗ 自动清理失败: {e}")
    
    def prune(self, force: bool = False) -> Dict:
        total_before = self.experience_collection.count()
        try:
            all_data = self.experience_collection.get(include=['metadatas'])
            now = datetime.now()
            ids_to_delete = []
            for i, meta in enumerate(all_data['metadatas']):
                doc_id = all_data['ids'][i]
                access_count = meta.get('access_count', 0)
                if not force and access_count >= self.PROMOTE_ACCESS_COUNT:
                    continue
                created_str = meta.get('created_time', meta.get('date', ''))
                try:
                    if ' ' in created_str:
                        created = datetime.strptime(created_str, "%Y-%m-%d %H:%M")
                    else:
                        created = datetime.strptime(created_str, "%Y-%m-%d")
                    age_days = (now - created).days
                except (ValueError, TypeError):
                    age_days = 0
                last_accessed_str = meta.get('last_accessed', created_str)
                try:
                    if ' ' in last_accessed_str:
                        last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d %H:%M")
                    else:
                        last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d")
                    inactive_days = (now - last_accessed).days
                except (ValueError, TypeError):
                    inactive_days = age_days
                should_delete = False
                if age_days > self.MAX_AGE_DAYS and inactive_days > self.MAX_INACTIVE_DAYS:
                    should_delete = True
                elif age_days > self.MAX_AGE_DAYS * 2:
                    should_delete = True
                if should_delete:
                    ids_to_delete.append(doc_id)
            if ids_to_delete:
                self.experience_collection.delete(ids=ids_to_delete)
            total_after = self.experience_collection.count()
            return {"before": total_before, "after": total_after,
                    "deleted": total_before - total_after, "force_mode": force}
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        total = self.experience_collection.count()
        if total == 0:
            return {"total_experiences": 0, "capacity": self.MAX_EXPERIENCES,
                    "usage": "0%", "promoted": 0, "stale": 0, "categories": {}}
        try:
            all_data = self.experience_collection.get(include=['metadatas'])
            now = datetime.now()
            promoted = 0; stale = 0; categories = {}
            for meta in all_data['metadatas']:
                cat = meta.get('category', 'default')
                if cat not in categories:
                    categories[cat] = {"count": 0, "limit": self.CATEGORY_LIMITS.get(cat, self.CATEGORY_LIMITS['default'])}
                categories[cat]["count"] += 1
                access_count = meta.get('access_count', 0)
                if access_count >= self.PROMOTE_ACCESS_COUNT:
                    promoted += 1
                last_accessed_str = meta.get('last_accessed', meta.get('date', ''))
                try:
                    if ' ' in last_accessed_str:
                        last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d %H:%M")
                    else:
                        last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d")
                    inactive_days = (now - last_accessed).days
                except (ValueError, TypeError):
                    inactive_days = 0
                if inactive_days > self.MAX_INACTIVE_DAYS:
                    stale += 1
            return {"total_experiences": total, "capacity": self.MAX_EXPERIENCES,
                    "usage": f"{total * 100 // self.MAX_EXPERIENCES}%",
                    "promoted": promoted, "stale": stale, "categories": categories,
                    "config": {"MAX_AGE_DAYS": self.MAX_AGE_DAYS,
                               "MAX_INACTIVE_DAYS": self.MAX_INACTIVE_DAYS,
                               "PROMOTE_ACCESS_COUNT": self.PROMOTE_ACCESS_COUNT}}
        except Exception as e:
            return {"error": str(e), "total_experiences": total, "categories": {}}
    
    # ==================== 统一查询接口 ====================
    
    def query(self, query_type: str, query_text: str) -> Dict:
        if query_type == "environment":
            return {"type": "environment", "data": self.environment, "source": "static"}
        elif query_type == "strategy":
            strategy = self.get_strategy(query_text)
            return {"type": "strategy", "query": query_text, "data": strategy,
                    "source": "static", "found": strategy is not None}
        elif query_type == "error":
            solution = self.get_error_solution(query_text)
            return {"type": "error_solution", "query": query_text, "data": solution,
                    "source": "static", "found": solution is not None}
        elif query_type == "experience":
            experiences = self.search_experience(query_text)
            return {"type": "experience", "query": query_text, "data": experiences,
                    "source": "dynamic", "count": len(experiences)}
        else:
            return {"type": "unknown", "error": f"Unknown query type: {query_type}"}
    
    # ==================== 初始化示例数据 ====================
    
    def initialize_sample_data(self):
        if not self.env_file.exists():
            self.environment = {
                "user_name": "华硕",
                "desktop_path": "C:\\Users\\华硕\\Desktop",
                "documents_path": "C:\\Users\\华硕\\Documents",
                "downloads_path": "C:\\Users\\华硕\\Downloads",
                "screen_resolution": "1920x1080",
                "python_path": "F:\\tools\\python.exe",
                "python_version": "3.11",
                "installed_apps": {
                    "记事本": "notepad", "WPS Office": "wps",
                    "VS Code": "code", "Chrome": "chrome", "Git": "git"
                },
                "common_paths": {
                    "transformer_project": "F:\\transformer",
                    "tools_directory": "F:\\tools"
                }
            }
            self._save_json(self.env_file, self.environment)
            print("✓ 已初始化环境信息")
        
        if not self.strategy_file.exists():
            self.strategies = {
                "创建文件": {
                    "best": "run_command",
                    "command": "echo. > \"{desktop_path}\\{filename}\"",
                    "alternative": "GUI右键新建（不推荐，容易失败）",
                    "warning": "不要用 echo 写中文，会乱码"
                },
                "启动应用": {
                    "best": "run_command", "command": "start {app_name}",
                    "alternative": "Win键搜索→Enter",
                    "warning": "应用名要准确，可以用 where 命令确认路径"
                },
                "打开记事本": {
                    "best": "run_command", "command": "start notepad",
                    "alternative": "Win键搜索→输入'记事本'→Enter",
                    "steps": ["Win键", "type_text '记事本'", "press_key enter"]
                },
                "保存文件": {
                    "best": "hotkey", "keys": ["ctrl", "s"],
                    "warning": "第一次保存会弹出对话框，需要输入文件名和路径"
                },
                "关闭窗口": {
                    "best": "hotkey", "keys": ["alt", "f4"],
                    "warning": "会提示是否保存未保存的内容"
                }
            }
            self._save_json(self.strategy_file, self.strategies)
            print("✓ 已初始化操作策略")
        
        if not self.error_file.exists():
            self.error_patterns = {
                "UnicodeDecodeError": {
                    "category": "编码问题",
                    "cause": "文件编码与默认编码不匹配",
                    "actions": [
                        "设置环境变量 PYTHONUTF8=1",
                        "改用 GUI 方式输入中文（type_text 工具）",
                        "指定文件编码：open(file, encoding='utf-8')"
                    ]
                },
                "UnicodeEncodeError": {
                    "category": "编码问题",
                    "cause": "API Key 或参数包含中文字符",
                    "actions": ["检查 API Key 是否为纯 ASCII", "替换占位符为真实值"]
                },
                "Permission denied": {
                    "category": "权限问题",
                    "cause": "没有足够的权限执行操作",
                    "actions": ["以管理员身份运行", "检查文件是否被其他程序占用", "检查文件是否为只读"]
                },
                "拒绝访问": {
                    "category": "权限问题",
                    "cause": "Windows 权限限制",
                    "actions": ["右键以管理员身份运行", "检查 UAC 设置", "检查文件所有者"]
                },
                "ModuleNotFoundError": {
                    "category": "依赖问题",
                    "cause": "缺少 Python 包",
                    "actions": ["pip install 包名", "检查是否在正确的虚拟环境中", "检查 Python 版本是否兼容"]
                },
                "FileNotFoundError": {
                    "category": "路径问题",
                    "cause": "文件或路径不存在",
                    "actions": ["检查路径是否正确", "用 dir 命令确认文件是否存在", "检查是否有拼写错误"]
                }
            }
            self._save_json(self.error_file, self.error_patterns)
            print("✓ 已初始化错误模式")
        
        print(f"\n知识库初始化完成！数据存储在：{self.knowledge_dir}")
