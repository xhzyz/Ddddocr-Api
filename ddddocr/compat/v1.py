# coding=utf-8
"""
向后兼容性支持模块
提供与原始DdddOcr类完全兼容的接口
"""

from typing import Union, List, Optional, Dict, Any, Tuple
import pathlib
from PIL import Image

from ..core.ocr_engine import OCREngine
from ..core.detection_engine import DetectionEngine
from ..core.slide_engine import SlideEngine
from ..preprocessing.color_filter import ColorFilter
from ..utils.exceptions import DDDDOCRError
from ..utils.validators import validate_model_config


def _normalize_color_arguments(
    colors: Optional[List[str]],
    custom_ranges: Any,
) -> Tuple[Optional[List[str]], Any]:
    """Normalize both the refactored and legacy color-filter shapes.

    Older examples use ``{"light_blue": [(lower), (upper)]}`` and refer to
    ``light_blue`` from ``colors``.  The internal filter consumes a flat list
    of HSV pairs, so convert that form while retaining built-in preset names.
    """
    if not isinstance(custom_ranges, dict):
        return colors, custom_ranges

    custom_names = {str(name).lower() for name in custom_ranges}
    preset_colors = None if colors is None else [
        color for color in colors if str(color).lower() not in custom_names
    ]
    flattened = []
    selected = {str(color).lower() for color in colors} if colors else None
    for name, ranges in custom_ranges.items():
        if selected is not None and str(name).lower() not in selected:
            continue
        if not isinstance(ranges, (list, tuple)):
            continue
        # One range is represented as [lower, upper]; multiple ranges are a
        # list of such pairs.
        if len(ranges) == 2 and all(isinstance(item, (list, tuple)) and len(item) == 3 for item in ranges):
            flattened.append((ranges[0], ranges[1]))
        else:
            for item in ranges:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    flattened.append((item[0], item[1]))
    return preset_colors or None, flattened or None


class DdddOcr:
    """
    DDDDOCR主类 - 向后兼容版本
    
    这个类保持与原始DdddOcr类完全相同的接口，
    但内部使用新的模块化架构实现
    """
    
    def __init__(self, ocr: bool = True, det: bool = False, old: bool = False, beta: bool = False,
                 use_gpu: bool = False, device_id: int = 0, show_ad: bool = True, 
                 import_onnx_path: str = "", charsets_path: str = ""):
        """
        初始化DDDDOCR
        
        Args:
            ocr: 是否启用OCR功能
            det: 是否启用目标检测功能
            old: 是否使用旧版OCR模型
            beta: 是否使用beta版OCR模型
            use_gpu: 是否使用GPU
            device_id: GPU设备ID
            show_ad: 是否显示广告信息
            import_onnx_path: 自定义ONNX模型路径
            charsets_path: 自定义字符集路径
        """
        # 显示广告信息（保持原有行为）
        if show_ad:
            print("欢迎使用ddddocr，本项目专注带动行业内卷，个人博客:wenanzhe.com")
            print("训练数据支持来源于:http://146.56.204.113:19199/preview")
            print("爬虫框架feapder可快速一键接入，快速开启爬虫之旅：https://github.com/Boris-code/feapder")
            print("谷歌reCaptcha验证码 / hCaptcha验证码 / funCaptcha验证码商业级识别接口：https://yescaptcha.com/i/NSwk7i")
        
        # 兼容性处理：确保PIL有ANTIALIAS属性
        if not hasattr(Image, 'ANTIALIAS'):
            setattr(Image, 'ANTIALIAS', Image.LANCZOS)
        
        # 验证配置参数
        validate_model_config(ocr, det, old, beta, use_gpu, device_id)
        
        # 保存配置
        self.ocr_enabled = ocr
        self.det_enabled = det
        self.old = old
        self.beta = beta
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.import_onnx_path = import_onnx_path
        self.charsets_path = charsets_path
        
        # 初始化引擎
        self.ocr_engine: Optional[OCREngine] = None
        self.detection_engine: Optional[DetectionEngine] = None
        self.slide_engine: Optional[SlideEngine] = None
        
        # 根据配置初始化相应的引擎
        if det:
            # 目标检测模式
            self.det = True
            self.detection_engine = DetectionEngine(use_gpu, device_id)
        elif ocr or import_onnx_path:
            # OCR模式
            self.det = False
            self.ocr_engine = OCREngine(
                use_gpu=use_gpu,
                device_id=device_id,
                old=old,
                beta=beta,
                import_onnx_path=import_onnx_path,
                charsets_path=charsets_path
            )
        else:
            # 滑块模式
            self.det = False
            
        # 滑块引擎总是可用
        self.slide_engine = SlideEngine()

    def color_filter(self, pil_image: Image.Image, colors: Optional[List[str]], custom_ranges=None) -> Image.Image:
        """Backward-compatible public color-filter helper."""
        preset_colors, normalized_ranges = _normalize_color_arguments(colors, custom_ranges)
        return ColorFilter(colors=preset_colors, custom_ranges=normalized_ranges).filter_image(pil_image)

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        """Backward-compatible detection preprocessing helper."""
        helper = self.detection_engine or DetectionEngine.__new__(DetectionEngine)
        return helper.preproc(img, input_size, swap)

    def demo_postprocess(self, outputs, img_size, p6=False):
        helper = self.detection_engine or DetectionEngine.__new__(DetectionEngine)
        return helper.demo_postprocess(outputs, img_size, p6)

    def nms(self, boxes, scores, nms_thr):
        helper = self.detection_engine or DetectionEngine.__new__(DetectionEngine)
        return helper.nms(boxes, scores, nms_thr)

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        helper = self.detection_engine or DetectionEngine.__new__(DetectionEngine)
        return helper.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        helper = self.detection_engine or DetectionEngine.__new__(DetectionEngine)
        return helper.multiclass_nms(boxes, scores, nms_thr, score_thr)

    def get_bbox(self, image_bytes: bytes):
        if not self.detection_engine:
            raise DDDDOCRError("目标检测功能未初始化")
        return self.detection_engine.get_bbox(image_bytes)

    def get_target(self, img_bytes=None, **kwargs):
        if img_bytes is None:
            img_bytes = kwargs.pop("target_bytes", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if img_bytes is None:
            raise DDDDOCRError("滑块目标图片不能为空")
        if not self.slide_engine:
            raise DDDDOCRError("滑块功能未初始化")
        return self.slide_engine.get_target(img_bytes)
    
    def classification(self, img: Union[bytes, str, pathlib.PurePath, Image.Image], 
                      png_fix: bool = False, probability: bool = False,
                      color_filter_colors: Optional[List[str]] = None,
                      color_filter_custom_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
                      charset_range: Optional[Union[int, str, List[str]]] = None,
                      colors: Optional[List[str]] = None,
                      custom_color_ranges: Optional[Union[
                          List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
                          Dict[str, Any],
                      ]] = None) -> Union[str, Dict[str, Any]]:
        """
        OCR识别方法
        
        Args:
            img: 图片数据（bytes、str、pathlib.PurePath或PIL.Image）
            png_fix: 是否修复PNG透明背景问题
            probability: 是否返回概率信息
            color_filter_colors: 颜色过滤预设颜色列表，如 ['red', 'blue']
            color_filter_custom_ranges: 自定义HSV颜色范围列表，如 [((0,50,50), (10,255,255))]
            charset_range: 单次识别使用的字符集范围
            colors: ``color_filter_colors`` 的向后兼容别名
            custom_color_ranges: ``color_filter_custom_ranges`` 的向后兼容别名
        
        Returns:
            识别结果文本或包含概率信息的字典
            
        Raises:
            DDDDOCRError: 当功能未启用或识别失败时
        """
        if self.det:
            raise DDDDOCRError("当前识别类型为目标检测")
        
        if not self.ocr_engine:
            raise DDDDOCRError("OCR功能未初始化")
        
        if color_filter_colors is None:
            color_filter_colors = colors
        if color_filter_custom_ranges is None:
            color_filter_custom_ranges = custom_color_ranges
        color_filter_colors, color_filter_custom_ranges = _normalize_color_arguments(
            color_filter_colors,
            color_filter_custom_ranges,
        )

        predict_kwargs = {
            "image": img,
            "png_fix": png_fix,
            "probability": probability,
            "color_filter_colors": color_filter_colors,
            "color_filter_custom_ranges": color_filter_custom_ranges,
        }
        if charset_range is not None:
            predict_kwargs["charset_range"] = charset_range

        return self.ocr_engine.predict(
            **predict_kwargs,
        )
    
    def detection(self, img: Union[bytes, str, pathlib.PurePath, Image.Image] = None,
                  **kwargs) -> List[List[int]]:
        """
        目标检测方法
        
        Args:
            img: 图片数据
            
        Returns:
            检测到的边界框列表
            
        Raises:
            DDDDOCRError: 当功能未启用或检测失败时
        """
        if img is None:
            img = kwargs.pop("img_bytes", None)
            if img is None:
                img = kwargs.pop("img_base64", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if img is None:
            raise DDDDOCRError("检测图片不能为空")

        if not self.det:
            raise DDDDOCRError("当前识别类型为OCR")
        
        if not self.detection_engine:
            raise DDDDOCRError("目标检测功能未初始化")
        
        return self.detection_engine.predict(img)
    
    def slide_match(self, target_img: Union[bytes, str, pathlib.PurePath, Image.Image] = None,
                   background_img: Union[bytes, str, pathlib.PurePath, Image.Image] = None,
                   simple_target: bool = False, **kwargs) -> Dict[str, Any]:
        """
        滑块匹配方法
        
        Args:
            target_img: 滑块图片
            background_img: 背景图片
            simple_target: 是否为简单滑块
            
        Returns:
            匹配结果字典
            
        Raises:
            DDDDOCRError: 当匹配失败时
        """
        if target_img is None:
            target_img = kwargs.pop("target_bytes", None)
            if target_img is None:
                target_img = kwargs.pop("target_image", None)
        if background_img is None:
            background_img = kwargs.pop("background_bytes", None)
            if background_img is None:
                background_img = kwargs.pop("background_image", None)
        flag = kwargs.pop("flag", False)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if target_img is None or background_img is None:
            raise DDDDOCRError("滑块目标图和背景图不能为空")

        if not self.slide_engine:
            raise DDDDOCRError("滑块功能未初始化")
        
        return self.slide_engine.slide_match(target_img, background_img, simple_target, flag)
    
    def slide_comparison(self, target_img: Union[bytes, str, pathlib.PurePath, Image.Image] = None,
                        background_img: Union[bytes, str, pathlib.PurePath, Image.Image] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        滑块比较方法
        
        Args:
            target_img: 带坑位的图片
            background_img: 完整背景图片
            
        Returns:
            比较结果字典
            
        Raises:
            DDDDOCRError: 当比较失败时
        """
        if target_img is None:
            target_img = kwargs.pop("target_bytes", None)
            if target_img is None:
                target_img = kwargs.pop("target_image", None)
        if background_img is None:
            background_img = kwargs.pop("background_bytes", None)
            if background_img is None:
                background_img = kwargs.pop("background_image", None)
        # ``flag`` was accepted by the old wrapper. The current comparison
        # engine does not use it, but accepting it preserves compatibility.
        kwargs.pop("flag", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if target_img is None or background_img is None:
            raise DDDDOCRError("滑块目标图和背景图不能为空")

        if not self.slide_engine:
            raise DDDDOCRError("滑块功能未初始化")
        
        return self.slide_engine.slide_comparison(target_img, background_img)
    
    def set_ranges(self, charset_range: Union[int, str, List[str]]) -> None:
        """
        设置字符集范围
        
        Args:
            charset_range: 字符集范围参数
            
        Raises:
            DDDDOCRError: 当OCR功能未启用时
        """
        if self.det:
            raise DDDDOCRError("目标检测模式不支持字符集设置")
        
        if not self.ocr_engine:
            raise DDDDOCRError("OCR功能未初始化")
        
        self.ocr_engine.set_charset_range(charset_range)
    
    def get_charset(self) -> List[str]:
        """
        获取字符集
        
        Returns:
            字符集列表
            
        Raises:
            DDDDOCRError: 当OCR功能未启用时
        """
        if self.det:
            raise DDDDOCRError("目标检测模式不支持字符集获取")
        
        if not self.ocr_engine:
            raise DDDDOCRError("OCR功能未初始化")
        
        return self.ocr_engine.get_charset()

    def clear_charset_range(self) -> None:
        """Clear a persistent charset restriction."""
        if self.det:
            raise DDDDOCRError("目标检测模式不支持字符集设置")
        if not self.ocr_engine:
            raise DDDDOCRError("OCR功能未初始化")
        self.ocr_engine.clear_charset_range()
    
    def switch_device(self, use_gpu: bool, device_id: int = 0) -> None:
        """
        切换计算设备
        
        Args:
            use_gpu: 是否使用GPU
            device_id: GPU设备ID
        """
        self.use_gpu = use_gpu
        self.device_id = device_id
        
        # 更新所有已初始化的引擎
        if self.ocr_engine:
            self.ocr_engine.switch_device(use_gpu, device_id)
        
        if self.detection_engine:
            self.detection_engine.switch_device(use_gpu, device_id)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'ocr_enabled': self.ocr_enabled,
            'det_enabled': self.det_enabled,
            'use_gpu': self.use_gpu,
            'device_id': self.device_id
        }
        
        if self.ocr_engine:
            info['ocr_model'] = self.ocr_engine.get_model_info()
        
        if self.detection_engine:
            info['detection_model'] = self.detection_engine.get_model_info()
        
        return info
    
    def cleanup(self) -> None:
        """清理所有资源"""
        ocr_engine = getattr(self, "ocr_engine", None)
        detection_engine = getattr(self, "detection_engine", None)
        slide_engine = getattr(self, "slide_engine", None)

        if ocr_engine:
            ocr_engine.cleanup()
        
        if detection_engine:
            detection_engine.cleanup()
        
        if slide_engine:
            slide_engine.cleanup()
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            # Destructors must never leak an exception during interpreter
            # shutdown or after a partially failed initialization.
            pass
    
    def __repr__(self) -> str:
        return f"DdddOcr(ocr={self.ocr_enabled}, det={self.det_enabled}, use_gpu={self.use_gpu})"
