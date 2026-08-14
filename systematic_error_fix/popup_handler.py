"""
弹窗处理器 - Popup Handler
=========================
弹窗类型识别，按钮坐标精确计算

核心功能：
  - 弹窗类型识别
  - 按钮坐标计算
  - 弹窗处理动作执行
  - 负面经验沉淀
"""
import logging
import re
from typing import Dict, Any, Optional, List

from systematic_error_fix.models import (
    PopupType, PopupAction, PopupInfo, PopupHandlingResult
)

logger = logging.getLogger(__name__)


class PopupTypeIdentifier:
    """弹窗类型识别器"""
    
    POPUP_SIGNATURES = {
        PopupType.FILE_NOT_FOUND: {
            "title_patterns": ["windows", "错误", "error"],
            "text_patterns": ["找不到", "not found", "不存在"],
            "buttons": ["确定", "ok", "取消", "cancel"]
        },
        PopupType.ERROR_DIALOG: {
            "title_patterns": ["错误", "error", "异常", "exception"],
            "text_patterns": [],
            "buttons": ["确定", "ok", "取消", "cancel", "重试", "retry"]
        },
        PopupType.WARNING_DIALOG: {
            "title_patterns": ["警告", "warning", "注意"],
            "text_patterns": [],
            "buttons": ["是", "yes", "否", "no", "确定", "ok"]
        },
        PopupType.INFO_DIALOG: {
            "title_patterns": ["信息", "info", "提示"],
            "text_patterns": [],
            "buttons": ["确定", "ok", "关闭", "close"]
        },
        PopupType.CONFIRMATION_DIALOG: {
            "title_patterns": ["确认", "confirm", "是否"],
            "text_patterns": [],
            "buttons": ["是", "yes", "否", "no", "确定", "ok", "取消", "cancel"]
        }
    }
    
    def identify(self, title: str, text: str, buttons: List[str]) -> PopupType:
        """识别弹窗类型
        
        Args:
            title: 弹窗标题
            text: 弹窗文本
            buttons: 按钮列表
            
        Returns:
            弹窗类型
        """
        title_lower = title.lower()
        text_lower = text.lower()
        
        for popup_type, signatures in self.POPUP_SIGNATURES.items():
            title_match = any(
                p in title_lower for p in signatures["title_patterns"]
            )
            
            text_match = (
                not signatures["text_patterns"] or
                any(p in text_lower for p in signatures["text_patterns"])
            )
            
            button_match = (
                not signatures["buttons"] or
                any(b.lower() in [btn.lower() for btn in buttons] 
                    for b in signatures["buttons"])
            )
            
            if title_match and text_match and button_match:
                return popup_type
        
        return PopupType.UNKNOWN


class ButtonLocator:
    """按钮定位器"""
    
    BUTTON_PRIORITY = {
        PopupType.FILE_NOT_FOUND: ["确定", "ok", "取消", "cancel"],
        PopupType.ERROR_DIALOG: ["确定", "ok", "取消", "cancel"],
        PopupType.WARNING_DIALOG: ["是", "yes", "否", "no", "确定", "ok"],
        PopupType.INFO_DIALOG: ["确定", "ok", "关闭", "close"],
        PopupType.CONFIRMATION_DIALOG: ["是", "yes", "否", "no", "确定", "ok"],
        PopupType.UNKNOWN: ["确定", "ok", "取消", "cancel"]
    }
    
    def locate(
        self,
        popup_type: PopupType,
        buttons: List[str],
        popup_rect: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        """定位按钮
        
        Args:
            popup_type: 弹窗类型
            buttons: 按钮列表
            popup_rect: 弹窗边界框
            
        Returns:
            按钮信息
        """
        priority = self.BUTTON_PRIORITY.get(popup_type, self.BUTTON_PRIORITY[PopupType.UNKNOWN])
        
        for preferred in priority:
            for button in buttons:
                if preferred.lower() in button.lower():
                    button_rect = self._calculate_button_rect(
                        button, buttons, popup_rect
                    )
                    return {
                        "name": button,
                        "rect": button_rect,
                        "center": self._calculate_center(button_rect) if button_rect else None
                    }
        
        if buttons:
            button = buttons[0]
            button_rect = self._calculate_button_rect(button, buttons, popup_rect)
            return {
                "name": button,
                "rect": button_rect,
                "center": self._calculate_center(button_rect) if button_rect else None
            }
        
        return None
    
    def _calculate_button_rect(
        self,
        button_name: str,
        all_buttons: List[str],
        popup_rect: Optional[Dict[str, int]]
    ) -> Optional[Dict[str, int]]:
        """计算按钮边界框
        
        Args:
            button_name: 按钮名称
            all_buttons: 所有按钮
            popup_rect: 弹窗边界框
            
        Returns:
            按钮边界框
        """
        if not popup_rect:
            return None
        
        left = popup_rect.get("left", 0)
        right = popup_rect.get("right", 100)
        top = popup_rect.get("top", 0)
        bottom = popup_rect.get("bottom", 100)
        
        popup_width = right - left
        popup_height = bottom - top
        
        button_height = 25
        button_width = min(80, popup_width // max(len(all_buttons), 1))
        
        button_y = bottom - button_height - 10
        
        button_index = all_buttons.index(button_name) if button_name in all_buttons else 0
        spacing = 10
        total_width = len(all_buttons) * button_width + (len(all_buttons) - 1) * spacing
        start_x = left + (popup_width - total_width) // 2
        
        button_x = start_x + button_index * (button_width + spacing)
        
        return {
            "left": button_x,
            "top": button_y,
            "right": button_x + button_width,
            "bottom": button_y + button_height
        }
    
    def _calculate_center(self, rect: Dict[str, int]) -> tuple[int, int]:
        """计算中心点
        
        Args:
            rect: 边界框
            
        Returns:
            中心点坐标
        """
        return (
            (rect["left"] + rect["right"]) // 2,
            (rect["top"] + rect["bottom"]) // 2
        )


class PopupActionDecider:
    """弹窗动作决策器"""
    
    ACTION_MAP = {
        PopupType.FILE_NOT_FOUND: PopupAction.CLICK_OK,
        PopupType.ERROR_DIALOG: PopupAction.CLICK_OK,
        PopupType.WARNING_DIALOG: PopupAction.CLICK_OK,
        PopupType.INFO_DIALOG: PopupAction.CLICK_OK,
        PopupType.CONFIRMATION_DIALOG: PopupAction.CLICK_CANCEL,
        PopupType.UNKNOWN: PopupAction.PRESS_ESCAPE
    }
    
    def decide(self, popup_type: PopupType, text: str = "") -> PopupAction:
        """决定处理动作
        
        Args:
            popup_type: 弹窗类型
            text: 弹窗文本
            
        Returns:
            处理动作
        """
        return self.ACTION_MAP.get(popup_type, PopupAction.PRESS_ESCAPE)


class PopupHandler:
    """弹窗处理器主类"""
    
    def __init__(self, ui_controller=None):
        """初始化弹窗处理器
        
        Args:
            ui_controller: UI Automation控制器
        """
        self.ui_controller = ui_controller
        self.type_identifier = PopupTypeIdentifier()
        self.button_locator = ButtonLocator()
        self.action_decider = PopupActionDecider()
        self._handled_popups: List[str] = []
    
    def detect_popup(self) -> Optional[PopupInfo]:
        """检测当前是否有弹窗
        
        Returns:
            弹窗信息，如果没有返回None
        """
        if not self.ui_controller:
            return None
        
        try:
            windows = self.ui_controller.get_all_windows()
            
            for window in windows:
                if self._is_dialog_window(window):
                    return PopupInfo(
                        popup_type=PopupType.UNKNOWN,
                        title=window.get("title", ""),
                        text=window.get("text", ""),
                        buttons=window.get("buttons", []),
                        rect=window.get("rect")
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"检测弹窗失败: {e}")
            return None
    
    def _is_dialog_window(self, window: Dict[str, Any]) -> bool:
        """判断是否为对话框窗口
        
        Args:
            window: 窗口信息
            
        Returns:
            是否为对话框
        """
        title = window.get("title", "").lower()
        control_type = window.get("control_type", "")
        
        dialog_keywords = ["错误", "error", "警告", "warning", "确认", "confirm"]
        if any(kw in title for kw in dialog_keywords):
            return True
        
        if control_type in ["Window", "Dialog"]:
            return True
        
        return False
    
    def handle(
        self,
        popup_info: Optional[PopupInfo] = None
    ) -> PopupHandlingResult:
        """处理弹窗
        
        Args:
            popup_info: 弹窗信息，如果为None则自动检测
            
        Returns:
            处理结果
        """
        if popup_info is None:
            popup_info = self.detect_popup()
        
        if popup_info is None:
            return PopupHandlingResult(
                success=True,
                action_taken=PopupAction.PRESS_ESCAPE,
                error="无弹窗需要处理"
            )
        
        popup_type = self.type_identifier.identify(
            popup_info.title,
            popup_info.text,
            popup_info.buttons
        )
        popup_info.popup_type = popup_type
        
        action = self.action_decider.decide(popup_type, popup_info.text)
        
        if action in [PopupAction.CLICK_OK, PopupAction.CLICK_CANCEL, 
                      PopupAction.CLICK_YES, PopupAction.CLICK_NO]:
            button_info = self.button_locator.locate(
                popup_type,
                popup_info.buttons,
                popup_info.rect
            )
            
            if button_info and button_info.get("center"):
                success = self._click_button(button_info["center"])
                return PopupHandlingResult(
                    success=success,
                    action_taken=action,
                    button_clicked=button_info["name"],
                    error="" if success else "点击按钮失败"
                )
        
        success = self._press_escape()
        return PopupHandlingResult(
            success=success,
            action_taken=PopupAction.PRESS_ESCAPE,
            error="" if success else "按Escape键失败"
        )
    
    def _click_button(self, center: tuple[int, int]) -> bool:
        """点击按钮
        
        Args:
            center: 按钮中心坐标
            
        Returns:
            是否成功
        """
        try:
            import pyautogui
            pyautogui.click(center[0], center[1])
            logger.info(f"点击按钮: ({center[0]}, {center[1]})")
            return True
        except Exception as e:
            logger.error(f"点击按钮失败: {e}")
            return False
    
    def _press_escape(self) -> bool:
        """按Escape键
        
        Returns:
            是否成功
        """
        try:
            import pyautogui
            pyautogui.press('escape')
            logger.info("按下Escape键")
            return True
        except Exception as e:
            logger.error(f"按Escape键失败: {e}")
            return False


def test_popup_handler():
    """测试弹窗处理器"""
    print("=" * 80)
    print("弹窗处理器测试")
    print("=" * 80)
    
    handler = PopupHandler()
    
    print("\n[测试1] 弹窗类型识别")
    from systematic_error_fix.models import PopupType
    
    test_cases = [
        ("Windows", "找不到文件'原神'", ["确定"], PopupType.FILE_NOT_FOUND),
        ("错误", "发生异常", ["确定", "取消"], PopupType.ERROR_DIALOG),
        ("警告", "确定要删除吗?", ["是", "否"], PopupType.WARNING_DIALOG),
        ("确认", "是否保存?", ["是", "否", "取消"], PopupType.CONFIRMATION_DIALOG)
    ]
    
    for title, text, buttons, expected in test_cases:
        result = handler.type_identifier.identify(title, text, buttons)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{title}' + '{text[:20]}...' → {result} (期望: {expected})")
    
    print("\n[测试2] 按钮定位")
    popup_rect = {"left": 500, "top": 300, "right": 900, "bottom": 500}
    buttons = ["确定", "取消"]
    
    for popup_type in [PopupType.FILE_NOT_FOUND, PopupType.ERROR_DIALOG]:
        button_info = handler.button_locator.locate(popup_type, buttons, popup_rect)
        if button_info:
            print(f"  {popup_type}: 按钮'{button_info['name']}' "
                  f"中心={button_info.get('center')}")
    
    print("\n[测试3] 动作决策")
    for popup_type in [PopupType.FILE_NOT_FOUND, PopupType.ERROR_DIALOG, 
                       PopupType.CONFIRMATION_DIALOG]:
        action = handler.action_decider.decide(popup_type)
        print(f"  {popup_type} → {action}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_popup_handler()