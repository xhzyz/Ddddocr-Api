# coding=utf-8
"""
滑块匹配引擎
提供滑块验证码的匹配和比较功能
"""

from typing import Union, Dict, Any, Tuple
import numpy as np
from PIL import Image

from .base import BaseEngine
from ..utils.image_io import load_image_from_input, image_to_numpy
from ..utils.exceptions import ImageProcessError, safe_import_opencv
from ..utils.validators import validate_image_input

# 安全导入OpenCV
cv2 = safe_import_opencv()


class SlideEngine(BaseEngine):
    """滑块匹配引擎"""
    
    def __init__(self):
        """
        初始化滑块引擎
        注意：滑块引擎不需要GPU和模型加载
        """
        # 不调用父类的__init__，因为不需要模型加载器
        self.is_initialized = True
    
    def initialize(self, **kwargs) -> None:
        """
        初始化滑块引擎
        滑块引擎不需要特殊初始化
        """
        self.is_initialized = True
    
    def predict(self, *args, **kwargs) -> Any:
        """
        预测方法的通用接口
        具体功能通过slide_match和slide_comparison方法实现
        """
        raise NotImplementedError("请使用slide_match或slide_comparison方法")
    
    def slide_match(self, target_image: Union[bytes, str, Image.Image], 
                   background_image: Union[bytes, str, Image.Image],
                   simple_target: bool = False,
                   flag: bool = False) -> Dict[str, Any]:
        """
        滑块匹配算法
        
        Args:
            target_image: 滑块图片
            background_image: 背景图片
            simple_target: 是否为简单滑块
            
        Returns:
            匹配结果字典，包含target坐标
            
        Raises:
            ImageProcessError: 当图像处理失败时
        """
        # 验证输入
        validate_image_input(target_image)
        validate_image_input(background_image)
        
        try:
            # 加载图像
            target_pil = load_image_from_input(target_image)
            background_pil = load_image_from_input(background_image)

            target_x = 0
            target_y = 0
            if not simple_target:
                try:
                    target_pil, target_x, target_y = self._crop_transparent_target(target_pil)
                except Exception:
                    if flag:
                        raise
                    # Preserve the original fallback: if an alpha target
                    # cannot be cropped, retry it as a simple target.
                    simple_target = True
            
            # 转换为numpy数组
            target_array = image_to_numpy(target_pil, 'RGB')
            background_array = image_to_numpy(background_pil, 'RGB')
            
            # 执行匹配
            result = self._perform_slide_match(target_array, background_array, simple_target)
            # These values are offsets of the visible slider inside the
            # original transparent target image, matching legacy ddddocr.
            result['target_x'] = int(target_x)
            result['target_y'] = int(target_y)
            
            return result
            
        except Exception as e:
            raise ImageProcessError(f"滑块匹配失败: {str(e)}") from e
    
    @staticmethod
    def _crop_transparent_target(image: Image.Image) -> Tuple[Image.Image, int, int]:
        """Crop a transparent slider image to its visible alpha bounds."""
        if "A" not in image.getbands():
            return image.convert("RGB"), 0, 0
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        visible_y, visible_x = np.where(alpha > 0)
        if visible_x.size == 0 or visible_y.size == 0:
            raise ValueError("target image has no visible pixels")
        left = int(visible_x.min())
        top = int(visible_y.min())
        right = int(visible_x.max()) + 1
        bottom = int(visible_y.max()) + 1
        cropped = rgba.crop((left, top, right, bottom))
        white = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, cropped).convert("RGB"), left, top

    def get_target(self, target_image: Union[bytes, str, Image.Image]) -> Tuple[Image.Image, int, int]:
        """Return the cropped visible target and its x/y offsets."""
        validate_image_input(target_image)
        try:
            image = load_image_from_input(target_image)
            return self._crop_transparent_target(image)
        except Exception as e:
            raise ImageProcessError(f"滑块目标裁剪失败: {str(e)}") from e

    def slide_comparison(self, target_image: Union[bytes, str, Image.Image],
                        background_image: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """
        滑块比较算法（用于带坑位的图片）
        
        Args:
            target_image: 带坑位的图片
            background_image: 完整背景图片
            
        Returns:
            比较结果字典，包含target坐标
            
        Raises:
            ImageProcessError: 当图像处理失败时
        """
        # 验证输入
        validate_image_input(target_image)
        validate_image_input(background_image)
        
        try:
            # 加载图像
            target_pil = load_image_from_input(target_image)
            background_pil = load_image_from_input(background_image)
            
            # 转换为numpy数组
            target_array = image_to_numpy(target_pil, 'RGB')
            background_array = image_to_numpy(background_pil, 'RGB')
            
            # 执行比较
            result = self._perform_slide_comparison(target_array, background_array)
            
            return result
            
        except Exception as e:
            raise ImageProcessError(f"滑块比较失败: {str(e)}") from e
    
    def _perform_slide_match(self, target: np.ndarray, background: np.ndarray, 
                           simple_target: bool) -> Dict[str, Any]:
        """
        执行滑块匹配
        
        Args:
            target: 滑块图像数组
            background: 背景图像数组
            simple_target: 是否为简单滑块
            
        Returns:
            匹配结果
        """
        try:
            # 转换为灰度图
            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
            
            if simple_target:
                # 简单滑块匹配
                result = self._simple_template_match(target_gray, background_gray)
            else:
                # 复杂滑块匹配（边缘检测）
                result = self._edge_based_match(target_gray, background_gray)
            
            return result
            
        except Exception as e:
            raise ImageProcessError(f"滑块匹配执行失败: {str(e)}") from e
    
    def _perform_slide_comparison(self, target: np.ndarray, background: np.ndarray) -> Dict[str, Any]:
        """
        执行滑块比较
        
        Args:
            target: 带坑位的图像数组
            background: 完整背景图像数组
            
        Returns:
            比较结果
        """
        try:
            # 计算图像差异
            diff = cv2.absdiff(target, background)
            
            # 转换为灰度图
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            
            # 形态学操作去噪
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'target': [0, 0]}
            
            # 找到最大的轮廓（假设是缺口）
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 返回差异区域左上角，保持旧版接口语义
            return {
                'target': [int(x), int(y)],
                'target_x': int(x),
                'target_y': int(y)
            }
            
        except Exception as e:
            raise ImageProcessError(f"滑块比较执行失败: {str(e)}") from e
    
    def _simple_template_match(self, target: np.ndarray, background: np.ndarray) -> Dict[str, Any]:
        """
        简单模板匹配
        
        Args:
            target: 滑块模板
            background: 背景图像
            
        Returns:
            匹配结果
        """
        try:
            # 模板匹配
            result = cv2.matchTemplate(background, target, cv2.TM_CCOEFF_NORMED)
            
            # 找到最佳匹配位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # 计算滑块中心位置
            if len(target.shape) == 3:
                target_h, target_w, _ = target.shape
            else:
                target_h, target_w = target.shape
            left = int(max_loc[0])
            top = int(max_loc[1])
            right = left + int(target_w)
            bottom = top + int(target_h)
            
            return {
                'target': [left, top, right, bottom],
                'target_x': 0,
                'target_y': 0,
                'confidence': float(max_val)
            }
            
        except Exception as e:
            raise ImageProcessError(f"简单模板匹配失败: {str(e)}") from e
    
    def _edge_based_match(self, target: np.ndarray, background: np.ndarray) -> Dict[str, Any]:
        """
        基于边缘检测的滑块匹配
        
        Args:
            target: 滑块图像
            background: 背景图像
            
        Returns:
            匹配结果
        """
        try:
            # 边缘检测
            target_edges = cv2.Canny(target, 50, 150)
            background_edges = cv2.Canny(background, 50, 150)
            
            # 模板匹配
            result = cv2.matchTemplate(background_edges, target_edges, cv2.TM_CCOEFF_NORMED)
            
            # 找到最佳匹配位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # 计算滑块中心位置
            if len(target.shape) == 3:
                target_h, target_w, _ = target.shape
            else:
                target_h, target_w = target.shape
            left = int(max_loc[0])
            top = int(max_loc[1])
            right = left + int(target_w)
            bottom = top + int(target_h)
            
            return {
                'target': [left, top, right, bottom],
                'target_x': 0,
                'target_y': 0,
                'confidence': float(max_val)
            }
            
        except Exception as e:
            raise ImageProcessError(f"边缘匹配失败: {str(e)}") from e
    
    def is_ready(self) -> bool:
        """
        检查引擎是否就绪
        滑块引擎总是就绪的
        
        Returns:
            总是返回True
        """
        return True
    
    def cleanup(self) -> None:
        """清理资源（滑块引擎无需清理）"""
        pass
    
    def __repr__(self) -> str:
        return "SlideEngine(ready=True)"
