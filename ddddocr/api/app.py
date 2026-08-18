#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production-oriented HTTP API for DdddOcr.

The project historically shipped multiple API implementations whose keyword
arguments no longer matched the refactored ``DdddOcr`` compatibility class.
This module is the single canonical API entry point used by both
``python -m ddddocr api`` and ``python -m ddddocr.api``.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import secrets
import threading
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from ddddocr import DdddOcr, DdddOcrInputError, InvalidImageError
from ddddocr.utils import (
    ALLOWED_IMAGE_FORMATS,
    MAX_IMAGE_BYTES as CORE_MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE as CORE_MAX_IMAGE_SIDE,
    DDDDOCRError,
    ImageProcessError,
    ModelLoadError,
)
from .models import (
    APIResponse as LegacyAPIResponse,
    DetectionRequest as LegacyDetectionRequest,
    DetectionResponse as LegacyDetectionResponse,
    InitializeRequest as LegacyInitializeRequest,
    OCRRequest as LegacyOCRRequest,
    OCRResponse as LegacyOCRResponse,
    StatusResponse as LegacyStatusResponse,
    SwitchModelRequest as LegacySwitchModelRequest,
    ToggleFeatureRequest as LegacyToggleFeatureRequest,
)


logger = logging.getLogger("ddddocr-api")
logging.basicConfig(
    level=os.environ.get("DDDDOCR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer in %s=%r; using %s", name, value, default)
        return default
    if parsed < minimum:
        logger.warning("%s must be >= %s; using %s", name, minimum, default)
        return default
    return parsed


def _package_version() -> str:
    try:
        return metadata.version("ddddocr")
    except metadata.PackageNotFoundError:
        return "1.6.1"


VERSION = _package_version()
MAX_IMAGE_BYTES = _env_int("DDDDOCR_MAX_IMAGE_BYTES", CORE_MAX_IMAGE_BYTES)
MAX_IMAGE_SIDE = _env_int("DDDDOCR_MAX_IMAGE_SIDE", CORE_MAX_IMAGE_SIDE)
MAX_BATCH_SIZE = _env_int("DDDDOCR_MAX_BATCH_SIZE", 32)
MAX_INSTANCES = _env_int("DDDDOCR_MAX_INSTANCES", 8)
INSTANCE_TTL_SECONDS = _env_int("DDDDOCR_INSTANCE_TTL_SECONDS", 3600)

DEFAULT_OLD = _env_bool("DDDDOCR_OLD", False)
DEFAULT_BETA = _env_bool("DDDDOCR_BETA", False)
PRELOAD_OCR = _env_bool("DDDDOCR_OCR", True)
PRELOAD_DET = _env_bool("DDDDOCR_DET", False)
DEFAULT_USE_GPU = _env_bool("DDDDOCR_USE_GPU", False)
DEFAULT_DEVICE_ID = _env_int("DDDDOCR_DEVICE_ID", 0, minimum=0)
DEFAULT_SHOW_AD = _env_bool("DDDDOCR_SHOW_AD", False)
DEFAULT_IMPORT_ONNX_PATH = os.environ.get("DDDDOCR_IMPORT_ONNX_PATH", "").strip()
DEFAULT_CHARSETS_PATH = os.environ.get("DDDDOCR_CHARSETS_PATH", "").strip()
API_KEY = os.environ.get("DDDDOCR_API_KEY", "").strip()

COLOR_PRESETS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "cyan",
    "black",
    "white",
    "gray",
}

OPENAPI_TAGS = [
    {"name": "系统", "description": "服务健康状态和运行配置。"},
    {"name": "OCR 识别", "description": "文字验证码识别，支持 Base64、文件上传和批量处理。"},
    {"name": "目标检测", "description": "调用 DdddOcr 目标检测模型返回边界框。"},
    {"name": "滑块识别", "description": "滑块模板匹配和两图差异比较。"},
    {"name": "模型管理", "description": "字符集、模型实例和缓存管理。"},
    {"name": "旧版兼容", "description": "兼容早期 HTTP API 客户端的接口。"},
    {"name": "MCP", "description": "供 MCP/Agent 客户端调用的兼容接口。"},
]


CharsetRange = Union[int, str, List[str]]
RawColorRanges = Union[List[List[List[int]]], Dict[str, Any]]


class OCRRequest(BaseModel):
    image: str = Field(..., description="图片的 Base64 字符串，也支持 Data URI")
    probability: bool = Field(False, description="是否返回逐字符概率信息")
    png_fix: bool = Field(False, description="是否将透明 PNG 铺到白色背景后再识别")
    colors: List[str] = Field(default_factory=list, description="预设 HSV 颜色名称列表")
    custom_color_ranges: Optional[RawColorRanges] = Field(
        None,
        description="自定义 HSV 范围：[[[h,s,v],[h,s,v]], ...]，也支持按名称组织的对象",
    )
    # Names used by the pre-1.6 API. They are accepted in addition to the
    # shorter ``colors``/``custom_color_ranges`` names.
    color_filter_colors: Optional[List[str]] = Field(None, description="旧版颜色过滤字段")
    color_filter_custom_ranges: Optional[RawColorRanges] = Field(
        None, description="旧版自定义颜色范围字段"
    )
    charset_range: Optional[CharsetRange] = Field(
        None,
        description="本次识别使用的字符范围；可传 0-7、字符串或字符列表",
    )


class OCRBatchRequest(BaseModel):
    images: List[OCRRequest] = Field(..., description="需要批量识别的图片请求列表")


class Base64ImageRequest(BaseModel):
    image: str = Field(..., description="图片的 Base64 字符串，也支持 Data URI")


class SlideMatchRequest(BaseModel):
    target_image: str = Field(..., description="滑块小图的 Base64 字符串")
    background_image: str = Field(..., description="背景大图的 Base64 字符串")
    simple_target: bool = Field(False, description="小图是否已经裁剪为简单目标")
    # When false, a crop failure falls back to simple-target mode. Setting
    # flag=true preserves the legacy strict behavior and surfaces the error.
    flag: bool = Field(False, description="透明区域裁剪失败时是否直接返回错误")


class SlideComparisonRequest(BaseModel):
    target_image: str = Field(..., description="带缺口图片的 Base64 字符串")
    background_image: str = Field(..., description="完整背景图片的 Base64 字符串")


class CharsetRangeRequest(BaseModel):
    # null resets the model instance and restores the complete charset.
    charset_range: Optional[CharsetRange] = Field(
        None, description="字符范围；传 null 恢复完整字符集"
    )


class TimedResponse(BaseModel):
    result: Any = Field(..., description="处理结果")
    processing_time: float = Field(..., description="处理耗时，单位为秒")


class BatchItemResponse(BaseModel):
    index: int = Field(..., description="图片在请求列表中的序号")
    result: Optional[Any] = Field(None, description="该图片的识别结果")
    error: Optional[str] = Field(None, description="该图片的错误信息")
    processing_time: float = Field(..., description="该图片处理耗时，单位为秒")


class BatchResponse(BaseModel):
    results: List[BatchItemResponse] = Field(..., description="每张图片的处理结果")
    processing_time: float = Field(..., description="整批请求总耗时，单位为秒")


class MCPRequestModel(BaseModel):
    method: str = Field(..., description="需要调用的 MCP 工具名称")
    params: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    id: Optional[Union[str, int]] = Field(None, description="请求 ID")


@dataclass(frozen=True)
class EngineConfig:
    mode: str
    old: bool = False
    beta: bool = False
    use_gpu: bool = False
    device_id: int = 0
    import_onnx_path: str = ""
    charsets_path: str = ""


@dataclass
class ManagedEngine:
    engine: DdddOcr
    config: EngineConfig
    lock: threading.RLock
    created_at: float
    last_used: float


class InstanceLimitError(RuntimeError):
    pass


class EngineRegistry:
    def __init__(self) -> None:
        self._items: Dict[EngineConfig, ManagedEngine] = {}
        self._lock = threading.RLock()

    def get(self, config: EngineConfig) -> ManagedEngine:
        with self._lock:
            managed = self._items.get(config)
            if managed is not None:
                managed.last_used = time.time()
                return managed

            self.cleanup_idle(INSTANCE_TTL_SECONDS)
            if len(self._items) >= MAX_INSTANCES:
                raise InstanceLimitError(
                    f"OCR instance limit reached ({MAX_INSTANCES}); clean inactive instances first"
                )

            logger.info("Creating DdddOcr instance: %s", config)
            if config.mode == "ocr":
                engine = DdddOcr(
                    ocr=True,
                    det=False,
                    old=config.old,
                    beta=config.beta,
                    use_gpu=config.use_gpu,
                    device_id=config.device_id,
                    show_ad=DEFAULT_SHOW_AD,
                    import_onnx_path=config.import_onnx_path,
                    charsets_path=config.charsets_path,
                )
            elif config.mode == "det":
                engine = DdddOcr(
                    ocr=False,
                    det=True,
                    use_gpu=config.use_gpu,
                    device_id=config.device_id,
                    show_ad=DEFAULT_SHOW_AD,
                )
            elif config.mode == "slide":
                engine = DdddOcr(ocr=False, det=False, show_ad=DEFAULT_SHOW_AD)
            else:
                raise ValueError(f"Unsupported engine mode: {config.mode}")

            now = time.time()
            managed = ManagedEngine(
                engine=engine,
                config=config,
                lock=threading.RLock(),
                created_at=now,
                last_used=now,
            )
            self._items[config] = managed
            return managed

    def reset(self, config: EngineConfig) -> None:
        with self._lock:
            managed = self._items.pop(config, None)
        if managed is not None:
            self._cleanup_engine(managed)

    def cleanup_idle(self, max_idle_seconds: int = INSTANCE_TTL_SECONDS) -> int:
        now = time.time()
        removed: List[ManagedEngine] = []
        with self._lock:
            for config, managed in list(self._items.items()):
                if now - managed.last_used > max_idle_seconds:
                    removed.append(self._items.pop(config))
        for managed in removed:
            self._cleanup_engine(managed)
        return len(removed)

    def cleanup_all(self) -> int:
        with self._lock:
            removed = list(self._items.values())
            self._items.clear()
        for managed in removed:
            self._cleanup_engine(managed)
        return len(removed)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "config": _public_config(managed.config),
                    "created_at": managed.created_at,
                    "last_used": managed.last_used,
                }
                for managed in self._items.values()
            ]

    @staticmethod
    def _cleanup_engine(managed: ManagedEngine) -> None:
        try:
            with managed.lock:
                managed.engine.cleanup()
        except Exception:
            logger.exception("Failed to clean DdddOcr instance: %s", managed.config)


registry = EngineRegistry()


@dataclass
class LegacyRuntime:
    """State used by the original initialize/switch/status API.

    The modern endpoints are lazy and stateless from a caller's point of
    view.  Older clients explicitly initialized a service first, so retain a
    small compatibility state while sharing the same engine registry and
    locks instead of loading duplicate ONNX sessions.
    """

    ocr_config: Optional[EngineConfig] = None
    det_config: Optional[EngineConfig] = None
    slide_config: Optional[EngineConfig] = None
    enabled_features: set[str] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    start_time: float = field(default_factory=time.time)

    def reset(self) -> None:
        with self.lock:
            configs = {
                config
                for config in (self.ocr_config, self.det_config, self.slide_config)
                if config is not None
            }
            self.ocr_config = None
            self.det_config = None
            self.slide_config = None
            self.enabled_features.clear()
        for config in configs:
            registry.reset(config)


legacy_runtime = LegacyRuntime()


def _public_config(config: EngineConfig) -> Dict[str, Any]:
    value = asdict(config)
    # Avoid leaking arbitrary host paths from configuration endpoints.
    value["import_onnx_path"] = bool(config.import_onnx_path)
    value["charsets_path"] = bool(config.charsets_path)
    return value


def _providers() -> List[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def _validate_model_paths() -> None:
    if bool(DEFAULT_IMPORT_ONNX_PATH) != bool(DEFAULT_CHARSETS_PATH):
        raise HTTPException(
            status_code=500,
            detail="DDDDOCR_IMPORT_ONNX_PATH and DDDDOCR_CHARSETS_PATH must be configured together",
        )


def _ocr_config(
    old: bool = Query(DEFAULT_OLD, description="是否使用旧版 OCR 模型"),
    beta: bool = Query(DEFAULT_BETA, description="是否使用 Beta OCR 模型"),
    use_gpu: bool = Query(DEFAULT_USE_GPU, description="是否使用 GPU 推理"),
    device_id: int = Query(DEFAULT_DEVICE_ID, ge=0, description="GPU 设备编号"),
) -> EngineConfig:
    if old and beta:
        raise HTTPException(status_code=400, detail="old and beta cannot both be true")
    _validate_model_paths()
    # A custom model determines its own model/charset, so old and beta do not
    # create duplicate instances in that mode.
    if DEFAULT_IMPORT_ONNX_PATH:
        old = False
        beta = False
    return EngineConfig(
        mode="ocr",
        old=old,
        beta=beta,
        use_gpu=use_gpu,
        device_id=device_id,
        import_onnx_path=DEFAULT_IMPORT_ONNX_PATH,
        charsets_path=DEFAULT_CHARSETS_PATH,
    )


def _det_config(
    use_gpu: bool = Query(DEFAULT_USE_GPU, description="是否使用 GPU 推理"),
    device_id: int = Query(DEFAULT_DEVICE_ID, ge=0, description="GPU 设备编号"),
) -> EngineConfig:
    return EngineConfig(mode="det", use_gpu=use_gpu, device_id=device_id)


def _slide_config() -> EngineConfig:
    return EngineConfig(mode="slide")


def _legacy_ocr_config(request: LegacyInitializeRequest) -> EngineConfig:
    if request.device_id < 0:
        raise ValueError("device_id must be non-negative")
    if request.old and request.beta:
        raise ValueError("old and beta cannot both be true")
    import_path = (request.import_onnx_path or DEFAULT_IMPORT_ONNX_PATH or "").strip()
    charset_path = (request.charsets_path or DEFAULT_CHARSETS_PATH or "").strip()
    if bool(import_path) != bool(charset_path):
        raise ValueError("import_onnx_path and charsets_path must be configured together")
    old = request.old
    beta = request.beta
    if import_path:
        old = False
        beta = False
    return EngineConfig(
        mode="ocr",
        old=old,
        beta=beta,
        use_gpu=request.use_gpu,
        device_id=request.device_id,
        import_onnx_path=import_path,
        charsets_path=charset_path,
    )


def _legacy_status_response() -> LegacyStatusResponse:
    with legacy_runtime.lock:
        loaded_models = []
        if legacy_runtime.ocr_config is not None:
            loaded_models.append("ocr")
        if legacy_runtime.det_config is not None:
            loaded_models.append("detection")
        if legacy_runtime.slide_config is not None:
            loaded_models.append("slide")
        enabled_features = set(legacy_runtime.enabled_features)
    for item in registry.snapshot():
        mode = item["config"]["mode"]
        name = "detection" if mode == "det" else mode
        if name not in loaded_models:
            loaded_models.append(name)
        enabled_features.add(name)
    with legacy_runtime.lock:
        return LegacyStatusResponse(
            service_status="running",
            loaded_models=loaded_models,
            enabled_features=sorted(enabled_features),
            version=VERSION,
            uptime=time.time() - legacy_runtime.start_time,
        )


def _legacy_error_response(exc: Exception) -> LegacyAPIResponse:
    return LegacyAPIResponse(success=False, message=str(exc), data=None)


async def _require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="可选 API 密钥"),
    authorization: Optional[str] = Header(
        None, alias="Authorization", description="也可使用 Bearer API 密钥"
    ),
) -> None:
    if not API_KEY:
        return
    supplied = x_api_key or ""
    if not supplied and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied = token
    if not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _decode_base64_image(value: str, field_name: str = "image") -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    encoded = value.strip()
    if encoded.lower().startswith("data:"):
        _, separator, encoded = encoded.partition(",")
        if not separator:
            raise HTTPException(status_code=400, detail=f"{field_name} data URI is invalid")
    encoded = "".join(encoded.split())
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid Base64") from exc
    return _validate_image_bytes(decoded, field_name)


def _validate_image_bytes(data: bytes, field_name: str = "image") -> bytes:
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_IMAGE_BYTES} byte limit",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=f"Unsupported image format: {image_format or 'unknown'}",
                    )
                if max(width, height) > MAX_IMAGE_SIDE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{field_name} longest side exceeds {MAX_IMAGE_SIDE}px",
                    )
                image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image") from exc
    return data


def _normalize_colors(
    colors: List[str],
    custom_ranges: Optional[RawColorRanges] = None,
) -> Optional[List[str]]:
    if not isinstance(colors, list):
        raise HTTPException(status_code=400, detail="colors must be a list")
    custom_names = {
        str(name).strip().lower()
        for name in custom_ranges
    } if isinstance(custom_ranges, dict) else set()
    normalized: List[str] = []
    for color in colors:
        if not isinstance(color, str) or not color.strip():
            raise HTTPException(status_code=400, detail="colors must contain non-empty strings")
        value = color.strip().lower()
        if value in custom_names:
            # Custom names are represented by custom_color_ranges and should
            # not be passed to ColorFilter as built-in presets.
            continue
        if value not in COLOR_PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported color {color!r}; allowed values: {sorted(COLOR_PRESETS)}",
            )
        normalized.append(value)
    return normalized or None


def _normalize_custom_ranges(
    raw: Optional[RawColorRanges],
    selected_colors: Optional[List[str]] = None,
) -> Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        selected = {str(item).strip().lower() for item in selected_colors or []}
        candidates = []
        for name, value in raw.items():
            if selected and str(name).strip().lower() not in selected:
                continue
            # A named color may contain one HSV pair or a list of pairs.
            if isinstance(value, list) and len(value) == 2 and all(
                isinstance(item, (list, tuple)) and len(item) == 3 for item in value
            ):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend(value)
            else:
                candidates.append(value)
    else:
        candidates = raw
    if not isinstance(candidates, list):
        raise HTTPException(status_code=400, detail="custom_color_ranges must be a list or object")

    # Also accept one pair directly: [[h1, s1, v1], [h2, s2, v2]].
    if len(candidates) == 2 and all(
        isinstance(item, (list, tuple)) and len(item) == 3 for item in candidates
    ):
        candidates = [candidates]

    normalized: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for index, pair in enumerate(candidates):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"custom_color_ranges[{index}] must contain lower and upper HSV values",
            )
        lower, upper = pair
        if not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
            raise HTTPException(status_code=400, detail="HSV bounds must be arrays")
        if len(lower) != 3 or len(upper) != 3:
            raise HTTPException(status_code=400, detail="Each HSV bound must have 3 integers")
        lower_tuple: List[int] = []
        upper_tuple: List[int] = []
        for channel, (low, high) in enumerate(zip(lower, upper)):
            if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int) or not isinstance(high, int):
                raise HTTPException(status_code=400, detail="HSV values must be integers")
            maximum = 180 if channel == 0 else 255
            if not 0 <= low <= maximum or not 0 <= high <= maximum:
                raise HTTPException(
                    status_code=400,
                    detail=f"HSV channel {channel} must be within 0-{maximum}",
                )
            if low > high:
                raise HTTPException(status_code=400, detail="HSV lower bound cannot exceed upper bound")
            lower_tuple.append(low)
            upper_tuple.append(high)
        normalized.append((tuple(lower_tuple), tuple(upper_tuple)))
    return normalized or None


def _validate_charset_range(value: Optional[CharsetRange]) -> Optional[CharsetRange]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail="charset_range cannot be boolean")
    if isinstance(value, int):
        if not 0 <= value <= 7:
            raise HTTPException(status_code=400, detail="charset_range integer must be between 0 and 7")
        return value
    if isinstance(value, str):
        if not value:
            raise HTTPException(status_code=400, detail="charset_range string cannot be empty")
        return value
    if isinstance(value, list):
        if not value or any(not isinstance(item, str) or not item for item in value):
            raise HTTPException(status_code=400, detail="charset_range must contain non-empty strings")
        return value
    raise HTTPException(status_code=400, detail="Unsupported charset_range type")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


KNOWN_LIBRARY_ERRORS = (
    DdddOcrInputError,
    InvalidImageError,
    DDDDOCRError,
    ImageProcessError,
    ModelLoadError,
    ValueError,
    ValidationError,
)


def _raise_library_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, InstanceLimitError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, KNOWN_LIBRARY_ERRORS):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.exception("Unhandled DdddOcr API error")
    raise HTTPException(status_code=500, detail="DdddOcr processing failed") from exc


def _ocr_sync(config: EngineConfig, image_data: bytes, request: OCRRequest) -> Any:
    selected_colors = request.colors or request.color_filter_colors or []
    selected_ranges = (
        request.custom_color_ranges
        if request.custom_color_ranges is not None
        else request.color_filter_custom_ranges
    )
    colors = _normalize_colors(selected_colors, selected_ranges)
    custom_ranges = _normalize_custom_ranges(selected_ranges, selected_colors)
    charset_range = _validate_charset_range(request.charset_range)
    managed = registry.get(config)
    with managed.lock:
        managed.last_used = time.time()
        classification_kwargs = {
            "png_fix": request.png_fix,
            "probability": request.probability,
            "color_filter_colors": colors,
            "color_filter_custom_ranges": custom_ranges,
        }
        if charset_range is not None:
            classification_kwargs["charset_range"] = charset_range
        return managed.engine.classification(
            image_data,
            **classification_kwargs,
        )


def _det_sync(config: EngineConfig, image_data: bytes) -> Any:
    managed = registry.get(config)
    with managed.lock:
        managed.last_used = time.time()
        return managed.engine.detection(image_data)


def _slide_match_sync(
    target_data: bytes,
    background_data: bytes,
    simple_target: bool,
    flag: bool = False,
) -> Any:
    managed = registry.get(_slide_config())
    with managed.lock:
        managed.last_used = time.time()
        return managed.engine.slide_match(
            target_data,
            background_data,
            simple_target=simple_target,
            flag=flag,
        )


def _slide_comparison_sync(target_data: bytes, background_data: bytes) -> Any:
    managed = registry.get(_slide_config())
    with managed.lock:
        managed.last_used = time.time()
        return managed.engine.slide_comparison(target_data, background_data)


def _parse_form_bool(value: Union[bool, str], field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"{field_name} must be true or false")


def _parse_json_form(value: str, field_name: str, default: Any) -> Any:
    if value is None or not value.strip():
        return default
    import json

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON") from exc


def _parse_charset_form(value: str) -> Optional[CharsetRange]:
    if value is None or not value.strip() or value.strip().lower() == "null":
        return None
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith('"') or stripped.isdigit():
        return _parse_json_form(stripped, "charset_range", None)
    return stripped


def _schedule_cleanup(background_tasks: BackgroundTasks) -> None:
    background_tasks.add_task(registry.cleanup_idle, INSTANCE_TTL_SECONDS)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if PRELOAD_OCR:
                startup_ocr_config = _ocr_config(
                    old=DEFAULT_OLD,
                    beta=DEFAULT_BETA,
                    use_gpu=DEFAULT_USE_GPU,
                    device_id=DEFAULT_DEVICE_ID,
                )
                await run_in_threadpool(registry.get, startup_ocr_config)
            if PRELOAD_DET:
                startup_det_config = _det_config(
                    use_gpu=DEFAULT_USE_GPU,
                    device_id=DEFAULT_DEVICE_ID,
                )
                await run_in_threadpool(registry.get, startup_det_config)
            yield
        finally:
            await run_in_threadpool(registry.cleanup_all)
            await run_in_threadpool(legacy_runtime.reset)

    application = FastAPI(
        title="DdddOcr 中文 API",
        description=(
            "DdddOcr 的完整 HTTP 接口，支持 OCR 识别、目标检测、滑块识别、"
            "自定义模型、文件上传、批量调用和旧版接口兼容。"
        ),
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=OPENAPI_TAGS,
        swagger_ui_parameters={
            "docExpansion": "list",
            "defaultModelsExpandDepth": 1,
            "displayRequestDuration": True,
            "filter": True,
        },
        lifespan=lifespan,
    )

    cors_value = os.environ.get("DDDDOCR_CORS_ORIGINS", "").strip()
    if cors_value:
        origins = [item.strip() for item in cors_value.split(",") if item.strip()]
        allow_all = origins == ["*"]
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    protected = APIRouter(dependencies=[Depends(_require_api_key)])

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @application.get("/health", tags=["系统"], summary="健康检查")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": VERSION,
            "providers": _providers(),
            "active_instances": len(registry.snapshot()),
            "timestamp": time.time(),
        }

    @protected.post(
        "/ocr",
        response_model=TimedResponse,
        tags=["OCR 识别"],
        summary="Base64 图片 OCR 识别",
        description="提交 Base64 图片进行文字识别，也支持 Data URI。",
    )
    async def ocr(
        request: OCRRequest,
        background_tasks: BackgroundTasks,
        config: EngineConfig = Depends(_ocr_config),
    ) -> Dict[str, Any]:
        image_data = _decode_base64_image(request.image)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(_ocr_sync, config, image_data, request)
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/ocr/file",
        response_model=TimedResponse,
        tags=["OCR 识别"],
        summary="上传图片文件进行 OCR 识别",
    )
    async def ocr_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="需要识别的图片文件"),
        probability: Union[bool, str] = Form(False, description="是否返回概率信息"),
        png_fix: Union[bool, str] = Form(False, description="是否修复透明 PNG 背景"),
        colors: str = Form("[]", description="预设颜色 JSON 数组"),
        custom_color_ranges: str = Form("null", description="自定义 HSV 范围 JSON"),
        color_filter_colors: Optional[str] = Form(None, description="旧版颜色字段 JSON"),
        color_filter_custom_ranges: Optional[str] = Form(
            None, description="旧版自定义颜色范围 JSON"
        ),
        charset_range: str = Form("null", description="字符范围 JSON、整数或字符串"),
        config: EngineConfig = Depends(_ocr_config),
    ) -> Dict[str, Any]:
        try:
            # Read one byte beyond the limit so an oversized upload is rejected
            # without buffering an unbounded request in memory.
            contents = await file.read(MAX_IMAGE_BYTES + 1)
        finally:
            await file.close()
        image_data = _validate_image_bytes(contents)
        # Prefer the modern field names when supplied; otherwise accept the
        # legacy multipart names used by the original API.
        colors_value = color_filter_colors if color_filter_colors is not None else colors
        ranges_value = (
            color_filter_custom_ranges
            if color_filter_custom_ranges is not None
            else custom_color_ranges
        )
        request = OCRRequest(
            image="unused",
            probability=_parse_form_bool(probability, "probability"),
            png_fix=_parse_form_bool(png_fix, "png_fix"),
            colors=_parse_json_form(colors_value, "colors", []),
            custom_color_ranges=_parse_json_form(
                ranges_value,
                "custom_color_ranges",
                None,
            ),
            charset_range=_parse_charset_form(charset_range),
        )
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(_ocr_sync, config, image_data, request)
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/ocr/batch",
        response_model=BatchResponse,
        tags=["OCR 识别"],
        summary="批量 OCR 识别",
        description="一次提交多张 Base64 图片；单张图片失败不会中断整批请求。",
    )
    async def ocr_batch(
        request: OCRBatchRequest,
        background_tasks: BackgroundTasks,
        config: EngineConfig = Depends(_ocr_config),
    ) -> Dict[str, Any]:
        if not request.images:
            raise HTTPException(status_code=400, detail="images cannot be empty")
        if len(request.images) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size exceeds the limit of {MAX_BATCH_SIZE}",
            )
        batch_started = time.perf_counter()
        responses: List[Dict[str, Any]] = []
        for index, item in enumerate(request.images):
            item_started = time.perf_counter()
            try:
                image_data = _decode_base64_image(item.image, f"images[{index}].image")
                result = await run_in_threadpool(_ocr_sync, config, image_data, item)
                responses.append(
                    {
                        "index": index,
                        "result": _json_safe(result),
                        "processing_time": time.perf_counter() - item_started,
                    }
                )
            except HTTPException as exc:
                responses.append(
                    {
                        "index": index,
                        "error": str(exc.detail),
                        "processing_time": time.perf_counter() - item_started,
                    }
                )
            except Exception as exc:
                if isinstance(exc, HTTPException):
                    error = str(exc.detail)
                elif isinstance(exc, InstanceLimitError):
                    error = str(exc)
                elif isinstance(exc, KNOWN_LIBRARY_ERRORS):
                    error = str(exc)
                else:
                    logger.exception("Batch OCR item %s failed", index)
                    error = "DdddOcr processing failed"
                responses.append(
                    {
                        "index": index,
                        "error": error,
                        "processing_time": time.perf_counter() - item_started,
                    }
                )
        _schedule_cleanup(background_tasks)
        return {"results": responses, "processing_time": time.perf_counter() - batch_started}

    @protected.post(
        "/det",
        response_model=TimedResponse,
        tags=["目标检测"],
        summary="Base64 图片目标检测",
    )
    async def detection(
        request: Base64ImageRequest,
        background_tasks: BackgroundTasks,
        config: EngineConfig = Depends(_det_config),
    ) -> Dict[str, Any]:
        image_data = _decode_base64_image(request.image)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(_det_sync, config, image_data)
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/det/file",
        response_model=TimedResponse,
        tags=["目标检测"],
        summary="上传图片文件进行目标检测",
    )
    async def detection_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="需要检测的图片文件"),
        config: EngineConfig = Depends(_det_config),
    ) -> Dict[str, Any]:
        try:
            contents = await file.read(MAX_IMAGE_BYTES + 1)
        finally:
            await file.close()
        image_data = _validate_image_bytes(contents)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(_det_sync, config, image_data)
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/slide_match",
        response_model=TimedResponse,
        tags=["滑块识别"],
        summary="滑块模板匹配",
        description="在背景大图中查找滑块小图，返回匹配框、裁剪偏移和置信度。",
    )
    @protected.post("/slide-match", response_model=TimedResponse, include_in_schema=False)
    async def slide_match(
        request: SlideMatchRequest,
        background_tasks: BackgroundTasks,
    ) -> Dict[str, Any]:
        target_data = _decode_base64_image(request.target_image, "target_image")
        background_data = _decode_base64_image(request.background_image, "background_image")
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(
                _slide_match_sync,
                target_data,
                background_data,
                request.simple_target,
                request.flag,
            )
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/slide_comparison",
        response_model=TimedResponse,
        tags=["滑块识别"],
        summary="滑块两图差异比较",
        description="比较带缺口图片和完整背景图片，返回缺口位置。",
    )
    @protected.post("/slide-comparison", response_model=TimedResponse, include_in_schema=False)
    async def slide_comparison(
        request: SlideComparisonRequest,
        background_tasks: BackgroundTasks,
    ) -> Dict[str, Any]:
        target_data = _decode_base64_image(request.target_image, "target_image")
        background_data = _decode_base64_image(request.background_image, "background_image")
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(
                _slide_comparison_sync,
                target_data,
                background_data,
            )
        except Exception as exc:
            _raise_library_error(exc)
        _schedule_cleanup(background_tasks)
        return {"result": _json_safe(result), "processing_time": time.perf_counter() - started}

    @protected.post(
        "/set_charset_range",
        tags=["模型管理"],
        summary="设置或重置字符范围",
    )
    @protected.post("/set-charset-range", include_in_schema=False)
    async def set_charset_range(
        request: CharsetRangeRequest,
        config: EngineConfig = Depends(_ocr_config),
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if request.charset_range is None:
            # Prefer clearing the range in-place so the loaded ONNX session
            # is retained. Fall back to resetting the instance for older
            # third-party engine implementations that do not expose the
            # helper yet.
            try:
                managed = await run_in_threadpool(registry.get, config)

                def clear_range() -> None:
                    with managed.lock:
                        managed.last_used = time.time()
                        clear = getattr(managed.engine, "clear_charset_range", None)
                        if clear is None:
                            raise AttributeError("clear_charset_range is unavailable")
                        clear()

                await run_in_threadpool(clear_range)
            except AttributeError:
                await run_in_threadpool(registry.reset, config)
            return {
                "result": "charset range reset",
                "charset_range": None,
                "processing_time": time.perf_counter() - started,
            }
        charset_range = _validate_charset_range(request.charset_range)
        try:
            managed = await run_in_threadpool(registry.get, config)

            def apply_range() -> None:
                with managed.lock:
                    managed.last_used = time.time()
                    managed.engine.set_ranges(charset_range)  # type: ignore[arg-type]

            await run_in_threadpool(apply_range)
        except Exception as exc:
            _raise_library_error(exc)
        return {
            "result": "charset range updated",
            "charset_range": charset_range,
            "processing_time": time.perf_counter() - started,
        }

    @protected.get("/charset", tags=["模型管理"], summary="查看当前字符集")
    async def get_charset(
        include_values: bool = Query(False, description="是否返回完整字符内容"),
        config: EngineConfig = Depends(_ocr_config),
    ) -> Dict[str, Any]:
        try:
            managed = await run_in_threadpool(registry.get, config)

            def read_charset() -> List[str]:
                with managed.lock:
                    managed.last_used = time.time()
                    return managed.engine.get_charset()

            charset = await run_in_threadpool(read_charset)
        except Exception as exc:
            _raise_library_error(exc)
        response: Dict[str, Any] = {"size": len(charset)}
        if include_values:
            response["charset"] = charset
        return response

    @protected.get("/model_info", tags=["模型管理"], summary="查看模型信息")
    @protected.get("/model-info", include_in_schema=False)
    async def model_info(
        mode: str = Query("ocr", pattern="^(ocr|det|slide)$", description="模型类型"),
        old: bool = Query(DEFAULT_OLD, description="是否使用旧版 OCR 模型"),
        beta: bool = Query(DEFAULT_BETA, description="是否使用 Beta OCR 模型"),
        use_gpu: bool = Query(DEFAULT_USE_GPU, description="是否使用 GPU 推理"),
        device_id: int = Query(DEFAULT_DEVICE_ID, ge=0, description="GPU 设备编号"),
    ) -> Dict[str, Any]:
        if mode == "ocr":
            config = _ocr_config(old=old, beta=beta, use_gpu=use_gpu, device_id=device_id)
        elif mode == "det":
            config = _det_config(use_gpu=use_gpu, device_id=device_id)
        else:
            config = _slide_config()
        try:
            managed = await run_in_threadpool(registry.get, config)

            def read_info() -> Dict[str, Any]:
                with managed.lock:
                    managed.last_used = time.time()
                    return managed.engine.get_model_info()

            info = await run_in_threadpool(read_info)
        except Exception as exc:
            _raise_library_error(exc)
        return {"config": _public_config(config), "model_info": _json_safe(info)}

    @protected.get("/config", tags=["系统"], summary="查看服务配置")
    async def current_config() -> Dict[str, Any]:
        return {
            "version": VERSION,
            "defaults": {
                "preload_ocr": PRELOAD_OCR,
                "preload_det": PRELOAD_DET,
                "old": DEFAULT_OLD,
                "beta": DEFAULT_BETA,
                "use_gpu": DEFAULT_USE_GPU,
                "device_id": DEFAULT_DEVICE_ID,
                "custom_model": bool(DEFAULT_IMPORT_ONNX_PATH),
                "custom_charset": bool(DEFAULT_CHARSETS_PATH),
            },
            "limits": {
                "max_image_bytes": MAX_IMAGE_BYTES,
                "max_image_side": MAX_IMAGE_SIDE,
                "max_batch_size": MAX_BATCH_SIZE,
                "max_instances": MAX_INSTANCES,
                "instance_ttl_seconds": INSTANCE_TTL_SECONDS,
            },
            "security": {
                "api_key_enabled": bool(API_KEY),
                "cors_origins_configured": bool(os.environ.get("DDDDOCR_CORS_ORIGINS", "").strip()),
            },
            "providers": _providers(),
            "active_instances": registry.snapshot(),
        }

    @protected.get("/instances", tags=["模型管理"], summary="查看已缓存模型实例")
    async def instances() -> Dict[str, Any]:
        return {"instances": registry.snapshot()}

    @protected.post(
        "/instances/cleanup",
        tags=["模型管理"],
        summary="清理模型实例缓存",
    )
    async def cleanup_instances(
        all_instances: bool = Query(False, alias="all", description="是否清理全部实例"),
        max_idle_seconds: int = Query(
            INSTANCE_TTL_SECONDS, ge=1, description="清理闲置超过该秒数的实例"
        ),
    ) -> Dict[str, Any]:
        if all_instances:
            removed = await run_in_threadpool(registry.cleanup_all)
        else:
            removed = await run_in_threadpool(registry.cleanup_idle, max_idle_seconds)
        return {"removed": removed, "remaining": len(registry.snapshot())}

    # ------------------------------------------------------------------
    # Legacy management endpoints
    # ------------------------------------------------------------------
    # These routes were present in the original API implementation. Keep
    # them as compatibility shims while using the same registry as the
    # canonical endpoints above, so callers are not forced to rewrite an
    # existing integration after upgrading.

    @protected.post(
        "/initialize",
        response_model=LegacyAPIResponse,
        tags=["旧版兼容"],
        summary="初始化旧版客户端所需模型",
    )
    async def legacy_initialize(request: LegacyInitializeRequest) -> LegacyAPIResponse:
        try:
            if request.device_id < 0:
                raise ValueError("device_id must be non-negative")
            ocr_config = _legacy_ocr_config(request) if request.ocr else None
            det_config = (
                EngineConfig(mode="det", use_gpu=request.use_gpu, device_id=request.device_id)
                if request.det
                else None
            )
            slide_config = _slide_config()

            configs_to_load = [config for config in (ocr_config, det_config, slide_config) if config]
            for config in configs_to_load:
                await run_in_threadpool(registry.get, config)

            with legacy_runtime.lock:
                old_configs = {
                    config
                    for config in (
                        legacy_runtime.ocr_config,
                        legacy_runtime.det_config,
                        legacy_runtime.slide_config,
                    )
                    if config is not None
                }
                legacy_runtime.ocr_config = ocr_config
                legacy_runtime.det_config = det_config
                legacy_runtime.slide_config = slide_config
                legacy_runtime.enabled_features = {
                    feature
                    for feature, enabled in (
                        ("ocr", request.ocr),
                        ("detection", request.det),
                        ("slide", True),
                    )
                    if enabled
                }

            for config in old_configs - set(configs_to_load):
                registry.reset(config)

            loaded_models = [
                name
                for name, enabled in (
                    ("ocr", request.ocr),
                    ("detection", request.det),
                    ("slide", True),
                )
                if enabled
            ]
            return LegacyAPIResponse(
                success=True,
                message="service initialized",
                data={"loaded_models": loaded_models, "message": "service initialized"},
            )
        except Exception as exc:
            logger.exception("Legacy initialize failed")
            return _legacy_error_response(exc)

    @protected.post(
        "/switch-model",
        response_model=LegacyAPIResponse,
        tags=["旧版兼容"],
        summary="切换旧版客户端模型",
    )
    async def legacy_switch_model(request: LegacySwitchModelRequest) -> LegacyAPIResponse:
        try:
            model_type = request.model_type.strip().lower()
            if request.device_id < 0:
                raise ValueError("device_id must be non-negative")
            if model_type in {"ocr", "ocr_old", "ocr_beta"}:
                init_request = LegacyInitializeRequest(
                    ocr=True,
                    old=model_type == "ocr_old",
                    beta=model_type == "ocr_beta",
                    use_gpu=request.use_gpu,
                    device_id=request.device_id,
                )
                config = _legacy_ocr_config(init_request)
                mode = "ocr"
            elif model_type == "det":
                config = EngineConfig(mode="det", use_gpu=request.use_gpu, device_id=request.device_id)
                mode = "det"
            elif model_type == "slide":
                config = _slide_config()
                mode = "slide"
            else:
                raise ValueError(f"unsupported model type: {request.model_type}")

            await run_in_threadpool(registry.get, config)
            with legacy_runtime.lock:
                old_config = getattr(legacy_runtime, f"{mode}_config", None)
                setattr(legacy_runtime, f"{mode}_config", config)
                legacy_runtime.enabled_features.add("detection" if mode == "det" else mode)
                if mode == "slide":
                    legacy_runtime.enabled_features.add("slide")
            if old_config is not None and old_config != config:
                registry.reset(old_config)
            return LegacyAPIResponse(
                success=True,
                message="model switched",
                data={"model_type": model_type, "message": "model switched"},
            )
        except Exception as exc:
            logger.exception("Legacy model switch failed")
            return _legacy_error_response(exc)

    @protected.post(
        "/toggle-feature",
        response_model=LegacyAPIResponse,
        tags=["旧版兼容"],
        summary="开启或关闭旧版功能",
    )
    async def legacy_toggle_feature(request: LegacyToggleFeatureRequest) -> LegacyAPIResponse:
        feature = request.feature.strip().lower()
        with legacy_runtime.lock:
            if request.enabled:
                legacy_runtime.enabled_features.add(feature)
            else:
                legacy_runtime.enabled_features.discard(feature)
        state = {"feature": feature, "enabled": request.enabled}
        return LegacyAPIResponse(
            success=True,
            message=f"feature {'enabled' if request.enabled else 'disabled'}",
            data=state,
        )

    @protected.post(
        "/detect",
        response_model=LegacyAPIResponse,
        tags=["旧版兼容"],
        summary="旧版目标检测接口",
    )
    @protected.post("/detection", response_model=LegacyAPIResponse, include_in_schema=False)
    async def legacy_detection(request: LegacyDetectionRequest) -> LegacyAPIResponse:
        try:
            with legacy_runtime.lock:
                config = legacy_runtime.det_config
                enabled = "detection" in legacy_runtime.enabled_features
            if config is None or not enabled:
                raise HTTPException(status_code=400, detail="detection is not initialized or enabled")
            image_data = _decode_base64_image(request.image)
            result = await run_in_threadpool(_det_sync, config, image_data)
            response_data = LegacyDetectionResponse(bboxes=result)
            data = (
                response_data.model_dump()
                if hasattr(response_data, "model_dump")
                else response_data.dict()
            )
            return LegacyAPIResponse(success=True, message="detection succeeded", data=data)
        except HTTPException:
            raise
        except Exception as exc:
            return _legacy_error_response(exc)

    @protected.get(
        "/status",
        response_model=LegacyStatusResponse,
        tags=["旧版兼容"],
        summary="查看旧版服务状态",
    )
    async def legacy_status() -> LegacyStatusResponse:
        return _legacy_status_response()

    application.include_router(protected)

    # The repository historically advertised an MCP adapter. Keep it on the
    # canonical application as a thin, correctly wired facade over the same
    # registry and validation code used by REST endpoints.
    mcp = APIRouter(prefix="/mcp", dependencies=[Depends(_require_api_key)])

    @mcp.get("/", tags=["MCP"], summary="查看 MCP 接口信息")
    async def mcp_info() -> Dict[str, Any]:
        return {
            "protocol": "MCP",
            "version": VERSION,
            "endpoints": {"capabilities": "/mcp/capabilities", "call": "/mcp/call"},
        }

    @mcp.get("/capabilities", tags=["MCP"], summary="查看 MCP 工具能力")
    async def mcp_capabilities() -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "ddddocr_initialize",
                    "description": "Preload requested model instances",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ocr": {"type": "boolean", "default": True},
                            "det": {"type": "boolean", "default": False},
                            "old": {"type": "boolean", "default": DEFAULT_OLD},
                            "beta": {"type": "boolean", "default": DEFAULT_BETA},
                            "use_gpu": {"type": "boolean", "default": DEFAULT_USE_GPU},
                            "device_id": {"type": "integer", "minimum": 0},
                            "import_onnx_path": {"type": "string"},
                            "charsets_path": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "ddddocr_ocr",
                    "description": "OCR text recognition",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "image": {"type": "string", "description": "Base64 image or data URI"},
                            "probability": {"type": "boolean", "default": False},
                            "png_fix": {"type": "boolean", "default": False},
                            "colors": {"type": "array", "items": {"type": "string"}},
                            "custom_color_ranges": {"type": ["array", "object", "null"]},
                            "charset_range": {
                                "anyOf": [
                                    {"type": "integer", "minimum": 0, "maximum": 7},
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["image"],
                    },
                },
                {
                    "name": "ddddocr_detection",
                    "description": "Object detection",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"image": {"type": "string"}},
                        "required": ["image"],
                    },
                },
                {
                    "name": "ddddocr_slide_match",
                    "description": "Slider template matching",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target_image": {"type": "string"},
                            "background_image": {"type": "string"},
                            "simple_target": {"type": "boolean", "default": False},
                            "flag": {"type": "boolean", "default": False},
                        },
                        "required": ["target_image", "background_image"],
                    },
                },
                {
                    "name": "ddddocr_slide_comparison",
                    "description": "Slider image comparison",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target_image": {"type": "string"},
                            "background_image": {"type": "string"},
                        },
                        "required": ["target_image", "background_image"],
                    },
                },
                {
                    "name": "ddddocr_status",
                    "description": "Service and model status",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ],
            "resources": [],
            "prompts": [],
        }

    @mcp.post("/call", tags=["MCP"], summary="调用 MCP 工具")
    async def mcp_call(request: MCPRequestModel) -> Dict[str, Any]:
        started = time.perf_counter()
        params = dict(request.params)
        try:
            method = request.method
            if method == "ddddocr_initialize":
                loaded: List[Dict[str, Any]] = []
                if _parse_form_bool(params.get("ocr", True), "ocr"):
                    init_request = LegacyInitializeRequest(
                        ocr=True,
                        old=_parse_form_bool(params.get("old", DEFAULT_OLD), "old"),
                        beta=_parse_form_bool(params.get("beta", DEFAULT_BETA), "beta"),
                        use_gpu=_parse_form_bool(params.get("use_gpu", DEFAULT_USE_GPU), "use_gpu"),
                        device_id=int(params.get("device_id", DEFAULT_DEVICE_ID)),
                        import_onnx_path=str(params.get("import_onnx_path", "")),
                        charsets_path=str(params.get("charsets_path", "")),
                    )
                    ocr_config = _legacy_ocr_config(init_request)
                    await run_in_threadpool(registry.get, ocr_config)
                    loaded.append(_public_config(ocr_config))
                if _parse_form_bool(params.get("det", False), "det"):
                    det_config = _det_config(
                        use_gpu=_parse_form_bool(params.get("use_gpu", DEFAULT_USE_GPU), "use_gpu"),
                        device_id=int(params.get("device_id", DEFAULT_DEVICE_ID)),
                    )
                    await run_in_threadpool(registry.get, det_config)
                    loaded.append(_public_config(det_config))
                await run_in_threadpool(registry.get, _slide_config())
                loaded.append(_public_config(_slide_config()))
                result = {"loaded": loaded}
            elif method == "ddddocr_status":
                result: Any = {
                    "status": "ok",
                    "version": VERSION,
                    "providers": _providers(),
                    "instances": registry.snapshot(),
                }
            elif method == "ddddocr_ocr":
                if "color_filter_colors" in params and "colors" not in params:
                    params["colors"] = params.pop("color_filter_colors")
                if "color_filter_custom_ranges" in params and "custom_color_ranges" not in params:
                    params["custom_color_ranges"] = params.pop("color_filter_custom_ranges")
                request_model = OCRRequest(**params)
                init_request = LegacyInitializeRequest(
                    ocr=True,
                    old=_parse_form_bool(params.get("old", DEFAULT_OLD), "old"),
                    beta=_parse_form_bool(params.get("beta", DEFAULT_BETA), "beta"),
                    use_gpu=_parse_form_bool(params.get("use_gpu", DEFAULT_USE_GPU), "use_gpu"),
                    device_id=int(params.get("device_id", DEFAULT_DEVICE_ID)),
                    import_onnx_path=str(params.get("import_onnx_path", "")),
                    charsets_path=str(params.get("charsets_path", "")),
                )
                config = _legacy_ocr_config(init_request)
                image_data = _decode_base64_image(request_model.image)
                result = await run_in_threadpool(_ocr_sync, config, image_data, request_model)
            elif method == "ddddocr_detection":
                image_data = _decode_base64_image(str(params.get("image", "")))
                config = EngineConfig(
                    mode="det",
                    use_gpu=_parse_form_bool(params.get("use_gpu", DEFAULT_USE_GPU), "use_gpu"),
                    device_id=int(params.get("device_id", DEFAULT_DEVICE_ID)),
                )
                result = await run_in_threadpool(_det_sync, config, image_data)
            elif method == "ddddocr_slide_match":
                target_data = _decode_base64_image(str(params.get("target_image", "")), "target_image")
                background_data = _decode_base64_image(
                    str(params.get("background_image", "")), "background_image"
                )
                result = await run_in_threadpool(
                    _slide_match_sync,
                    target_data,
                    background_data,
                    _parse_form_bool(params.get("simple_target", False), "simple_target"),
                    _parse_form_bool(params.get("flag", False), "flag"),
                )
            elif method == "ddddocr_slide_comparison":
                target_data = _decode_base64_image(str(params.get("target_image", "")), "target_image")
                background_data = _decode_base64_image(
                    str(params.get("background_image", "")), "background_image"
                )
                result = await run_in_threadpool(_slide_comparison_sync, target_data, background_data)
            else:
                raise ValueError(f"Unsupported MCP method: {method}")

            return {
                "result": _json_safe(result),
                "processing_time": time.perf_counter() - started,
                "id": request.id,
            }
        except HTTPException as exc:
            return {"error": {"code": exc.status_code, "message": str(exc.detail)}, "id": request.id}
        except Exception as exc:
            if isinstance(exc, KNOWN_LIBRARY_ERRORS):
                message = str(exc)
            else:
                logger.exception("MCP call failed")
                message = "DdddOcr processing failed"
            return {"error": {"code": -1, "message": message}, "id": request.id}

    application.include_router(mcp)

    return application


app = create_app()


def main() -> None:
    host = os.environ.get("DDDDOCR_HOST", "127.0.0.1")
    port = _env_int("DDDDOCR_PORT", 8000)
    workers = _env_int("DDDDOCR_WORKERS", 1)
    uvicorn.run("ddddocr.api:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
