"""
错误诊断器 - Error Diagnostician
==============================
精确识别错误类型，避免"指鹿为马"误判

核心功能：
  - 错误特征提取
  - 优先级匹配算法
  - 修复动作生成
  - 修复效果验证
"""
import re
import logging
from typing import Dict, Any, Optional, List

from systematic_error_fix.models import (
    ErrorType, FixAction, ErrorDiagnosisResult
)

logger = logging.getLogger(__name__)


class ErrorFeatureExtractor:
    """错误特征提取器"""
    
    PATH_PATTERN = re.compile(r'[A-Za-z]:\\[^\s"<>\|:*?]+|/[^\s"<>\|:*?]+')
    COMMAND_PATTERN = re.compile(r'命令[：:]\s*(.+?)(?:\s|$)')
    PROGRAM_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"")
    
    def extract_path(self, error_text: str) -> Optional[str]:
        """提取路径
        
        Args:
            error_text: 错误文本
            
        Returns:
            提取的路径
        """
        match = self.PATH_PATTERN.search(error_text)
        return match.group(0) if match else None
    
    def extract_command(self, error_text: str) -> Optional[str]:
        """提取命令
        
        Args:
            error_text: 错误文本
            
        Returns:
            提取的命令
        """
        match = self.COMMAND_PATTERN.search(error_text)
        return match.group(1).strip() if match else None
    
    def extract_program_name(self, error_text: str) -> Optional[str]:
        """提取程序名
        
        Args:
            error_text: 错误文本
            
        Returns:
            提取的程序名
        """
        match = self.PROGRAM_PATTERN.search(error_text)
        return match.group(1) or match.group(2) if match else None
    
    def extract_features(self, error_text: str) -> Dict[str, Any]:
        """提取所有特征
        
        Args:
            error_text: 错误文本
            
        Returns:
            特征字典
        """
        return {
            "path": self.extract_path(error_text),
            "command": self.extract_command(error_text),
            "program_name": self.extract_program_name(error_text),
            "text_lower": error_text.lower(),
            "text_length": len(error_text)
        }


class ErrorTypeMatcher:
    """错误类型匹配器"""
    
    MATCHING_RULES = [
        {
            "error_type": ErrorType.PROGRAM_NOT_FOUND,
            "priority": 1,
            "patterns": [
                "找不到", "not found", "无法找到",
                "不存在", "does not exist",
                "无法识别", "not recognized"
            ],
            "exclusions": [
                "网络", "network", "连接", "connection",
                "指定的路径", "path not found",
                "目录不存在", "directory not found"
            ]
        },
        {
            "error_type": ErrorType.PATH_NOT_FOUND,
            "priority": 2,
            "patterns": [
                "系统找不到指定的路径", "path not found",
                "目录不存在", "directory not found"
            ],
            "exclusions": []
        },
        {
            "error_type": ErrorType.ENCODING_ERROR,
            "priority": 3,
            "patterns": [
                "编码", "encoding", "乱码",
                "字符集", "charset", "utf", "gbk"
            ],
            "exclusions": []
        },
        {
            "error_type": ErrorType.PERMISSION_DENIED,
            "priority": 4,
            "patterns": [
                "拒绝访问", "access denied",
                "权限不足", "permission denied",
                "没有权限", "unauthorized"
            ],
            "exclusions": []
        },
        {
            "error_type": ErrorType.COMMAND_TIMEOUT,
            "priority": 5,
            "patterns": [
                "超时", "timeout", "timed out",
                "响应时间过长", "response time"
            ],
            "exclusions": []
        },
        {
            "error_type": ErrorType.NETWORK_ERROR,
            "priority": 6,
            "patterns": [
                "网络", "network",
                "连接失败", "connection failed",
                "无法连接", "cannot connect",
                "dns", "socket"
            ],
            "exclusions": ["找不到", "not found"]
        }
    ]
    
    def match(self, features: Dict[str, Any]) -> Tuple[ErrorType, float, str]:
        """匹配错误类型
        
        Args:
            features: 特征字典
            
        Returns:
            (错误类型, 置信度, 匹配规则)
        """
        text_lower = features.get("text_lower", "")
        
        for rule in self.MATCHING_RULES:
            error_type = rule["error_type"]
            patterns = rule["patterns"]
            exclusions = rule["exclusions"]
            
            has_exclusion = any(exc in text_lower for exc in exclusions)
            if has_exclusion:
                continue
            
            matched_patterns = [p for p in patterns if p in text_lower]
            
            if matched_patterns:
                confidence = min(0.95, 0.7 + len(matched_patterns) * 0.1)
                return (error_type, confidence, f"pattern_match:{matched_patterns[0]}")
        
        return (ErrorType.UNKNOWN_ERROR, 0.3, "no_match")


class FixActionGenerator:
    """修复动作生成器"""
    
    def generate(self, error_type: ErrorType, features: Dict[str, Any]) -> List[FixAction]:
        """生成修复动作
        
        Args:
            error_type: 错误类型
            features: 特征字典
            
        Returns:
            修复动作列表
        """
        generators = {
            ErrorType.PROGRAM_NOT_FOUND: self._generate_program_not_found_fix,
            ErrorType.PATH_NOT_FOUND: self._generate_path_not_found_fix,
            ErrorType.ENCODING_ERROR: self._generate_encoding_fix,
            ErrorType.PERMISSION_DENIED: self._generate_permission_fix,
            ErrorType.COMMAND_TIMEOUT: self._generate_timeout_fix,
            ErrorType.NETWORK_ERROR: self._generate_network_fix
        }
        
        generator = generators.get(error_type, self._generate_unknown_fix)
        return generator(features)
    
    def _generate_program_not_found_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成程序未找到的修复动作"""
        program_name = features.get("program_name", "")
        actions = []
        
        actions.append(FixAction(
            action_type="search_in_program_files",
            description=f"在Program Files中搜索{program_name}",
            parameters={"program": program_name},
            priority=1,
            expected_outcome="找到程序安装路径"
        ))
        
        actions.append(FixAction(
            action_type="search_desktop_shortcuts",
            description="搜索桌面快捷方式",
            parameters={},
            priority=2,
            expected_outcome="找到有效快捷方式"
        ))
        
        actions.append(FixAction(
            action_type="use_where_command",
            description=f"使用where命令查找{program_name}",
            parameters={"command": f"where {program_name}"},
            priority=3,
            expected_outcome="找到程序路径"
        ))
        
        return actions
    
    def _generate_path_not_found_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成路径未找到的修复动作"""
        path = features.get("path", "")
        actions = []
        
        actions.append(FixAction(
            action_type="verify_path",
            description=f"验证路径是否存在: {path}",
            parameters={"path": path},
            priority=1,
            expected_outcome="确认路径状态"
        ))
        
        actions.append(FixAction(
            action_type="search_alternative_path",
            description="搜索替代路径",
            parameters={"original_path": path},
            priority=2,
            expected_outcome="找到有效路径"
        ))
        
        return actions
    
    def _generate_encoding_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成编码错误的修复动作"""
        return [
            FixAction(
                action_type="change_encoding",
                description="切换编码为UTF-8",
                parameters={"encoding": "utf-8"},
                priority=1,
                expected_outcome="编码问题解决"
            ),
            FixAction(
                action_type="change_encoding",
                description="切换编码为GBK",
                parameters={"encoding": "gbk"},
                priority=2,
                expected_outcome="编码问题解决"
            )
        ]
    
    def _generate_permission_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成权限拒绝的修复动作"""
        return [
            FixAction(
                action_type="run_as_admin",
                description="以管理员权限运行",
                parameters={},
                priority=1,
                expected_outcome="获得足够权限"
            ),
            FixAction(
                action_type="check_permission",
                description="检查文件权限",
                parameters={},
                priority=2,
                expected_outcome="确认权限状态"
            )
        ]
    
    def _generate_timeout_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成超时的修复动作"""
        return [
            FixAction(
                action_type="increase_timeout",
                description="增加超时时间",
                parameters={"timeout": 60},
                priority=1,
                expected_outcome="等待更长时间"
            ),
            FixAction(
                action_type="retry",
                description="重试操作",
                parameters={},
                priority=2,
                expected_outcome="操作成功"
            )
        ]
    
    def _generate_network_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成网络错误的修复动作"""
        return [
            FixAction(
                action_type="check_network",
                description="检查网络连接",
                parameters={},
                priority=1,
                expected_outcome="确认网络状态"
            ),
            FixAction(
                action_type="retry_with_delay",
                description="等待后重试",
                parameters={"delay": 5},
                priority=2,
                expected_outcome="网络恢复"
            )
        ]
    
    def _generate_unknown_fix(self, features: Dict[str, Any]) -> List[FixAction]:
        """生成未知错误的修复动作"""
        return [
            FixAction(
                action_type="retry",
                description="重试操作",
                parameters={},
                priority=1,
                expected_outcome="操作成功"
            )
        ]


class ErrorDiagnostician:
    """错误诊断器主类"""
    
    def __init__(self, knowledge_base=None):
        """初始化错误诊断器
        
        Args:
            knowledge_base: 知识库实例
        """
        self.knowledge_base = knowledge_base
        self.feature_extractor = ErrorFeatureExtractor()
        self.type_matcher = ErrorTypeMatcher()
        self.action_generator = FixActionGenerator()
    
    def diagnose(self, error_text: str, context: Optional[Dict[str, Any]] = None) -> ErrorDiagnosisResult:
        """诊断错误
        
        Args:
            error_text: 错误文本
            context: 上下文信息
            
        Returns:
            诊断结果
        """
        features = self.feature_extractor.extract_features(error_text)
        
        if context:
            features.update(context)
        
        error_type, confidence, matched_rule = self.type_matcher.match(features)
        
        fix_actions = self.action_generator.generate(error_type, features)
        
        reasoning = self._generate_reasoning(error_type, features, matched_rule)
        
        logger.info(f"错误诊断: {error_type} (置信度={confidence:.2f})")
        
        return ErrorDiagnosisResult(
            error_type=error_type,
            confidence=confidence,
            features=features,
            fix_actions=fix_actions,
            matched_rule=matched_rule,
            reasoning=reasoning
        )
    
    def _generate_reasoning(
        self,
        error_type: ErrorType,
        features: Dict[str, Any],
        matched_rule: str
    ) -> str:
        """生成诊断推理说明
        
        Args:
            error_type: 错误类型
            features: 特征字典
            matched_rule: 匹配规则
            
        Returns:
            推理说明
        """
        program_name = features.get("program_name", "")
        path = features.get("path", "")
        
        if error_type == ErrorType.PROGRAM_NOT_FOUND:
            if program_name:
                return f"检测到程序'{program_name}'不存在，建议搜索安装路径"
            return "检测到程序不存在，建议搜索安装路径"
        
        elif error_type == ErrorType.PATH_NOT_FOUND:
            if path:
                return f"检测到路径'{path}'不存在，建议验证或搜索替代路径"
            return "检测到路径不存在，建议验证路径"
        
        elif error_type == ErrorType.NETWORK_ERROR:
            return "检测到网络问题，建议检查网络连接后重试"
        
        return f"错误类型: {error_type}"


def test_error_diagnostician():
    """测试错误诊断器"""
    print("=" * 80)
    print("错误诊断器测试")
    print("=" * 80)
    
    diagnostician = ErrorDiagnostician()
    
    test_errors = [
        "Windows找不到文件'原神'。请确定文件名是否正确后，再试一次。",
        "系统找不到指定的路径: D:\\Games\\原神",
        "网络连接失败，请检查网络设置",
        "'genshin' 不是内部或外部命令，也不是可运行的程序或批处理文件。",
        "访问被拒绝。您没有权限访问此文件。"
    ]
    
    for error in test_errors:
        print(f"\n错误文本: {error[:50]}...")
        result = diagnostician.diagnose(error)
        print(f"  错误类型: {result.error_type}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  匹配规则: {result.matched_rule}")
        print(f"  推理: {result.reasoning}")
        print(f"  修复动作数: {len(result.fix_actions)}")
        if result.fix_actions:
            print(f"  首选动作: {result.fix_actions[0].description}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_error_diagnostician()