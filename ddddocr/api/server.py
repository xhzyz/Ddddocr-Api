# coding=utf-8
"""Compatibility imports for code using ``ddddocr.api.server``.

The HTTP application itself lives in :mod:`ddddocr.api.app`.  A small service
facade is retained for callers that used the pre-1.6 ``DDDDOCRService`` class.
"""

import time

import ddddocr
from .app import app, create_app
from .models import (
    InitializeRequest,
    StatusResponse,
    SwitchModelRequest,
    ToggleFeatureRequest,
)


class DDDDOCRService:
    """Backward-compatible in-process service facade."""

    def __init__(self):
        self.ocr_instance = None
        self.det_instance = None
        self.slide_instance = None
        self.enabled_features = set()
        self.start_time = time.time()
        self.version = "1.6.1"

    def initialize(self, config: InitializeRequest):
        self.cleanup()
        self.enabled_features.clear()
        if config.ocr:
            self.ocr_instance = ddddocr.DdddOcr(
                ocr=True,
                det=False,
                old=config.old,
                beta=config.beta,
                use_gpu=config.use_gpu,
                device_id=config.device_id,
                show_ad=False,
                import_onnx_path=config.import_onnx_path,
                charsets_path=config.charsets_path,
            )
            self.enabled_features.add("ocr")
        if config.det:
            self.det_instance = ddddocr.DdddOcr(
                ocr=False,
                det=True,
                use_gpu=config.use_gpu,
                device_id=config.device_id,
                show_ad=False,
            )
            self.enabled_features.add("detection")
        self.slide_instance = ddddocr.DdddOcr(ocr=False, det=False, show_ad=False)
        self.enabled_features.add("slide")
        return {"loaded_models": list(self.enabled_features), "message": "service initialized"}

    def switch_model(self, config: SwitchModelRequest):
        if config.model_type == "ocr":
            self.ocr_instance = ddddocr.DdddOcr(
                ocr=True, det=False, use_gpu=config.use_gpu, device_id=config.device_id, show_ad=False
            )
            self.enabled_features.add("ocr")
        elif config.model_type == "ocr_old":
            self.ocr_instance = ddddocr.DdddOcr(
                ocr=True, det=False, old=True, use_gpu=config.use_gpu, device_id=config.device_id, show_ad=False
            )
            self.enabled_features.add("ocr")
        elif config.model_type == "ocr_beta":
            self.ocr_instance = ddddocr.DdddOcr(
                ocr=True, det=False, beta=True, use_gpu=config.use_gpu, device_id=config.device_id, show_ad=False
            )
            self.enabled_features.add("ocr")
        elif config.model_type == "det":
            self.det_instance = ddddocr.DdddOcr(
                ocr=False, det=True, use_gpu=config.use_gpu, device_id=config.device_id, show_ad=False
            )
            self.enabled_features.add("detection")
        else:
            raise ValueError(f"unsupported model type: {config.model_type}")
        return {"model_type": config.model_type, "message": "model switched"}

    def toggle_feature(self, config: ToggleFeatureRequest):
        if config.enabled:
            self.enabled_features.add(config.feature)
        else:
            self.enabled_features.discard(config.feature)
        return {"feature": config.feature, "enabled": config.enabled}

    def get_status(self) -> StatusResponse:
        loaded = []
        if self.ocr_instance:
            loaded.append("ocr")
        if self.det_instance:
            loaded.append("detection")
        if self.slide_instance:
            loaded.append("slide")
        return StatusResponse(
            service_status="running",
            loaded_models=loaded,
            enabled_features=list(self.enabled_features),
            version=self.version,
            uptime=time.time() - self.start_time,
        )

    def cleanup(self):
        for instance in (self.ocr_instance, self.det_instance, self.slide_instance):
            if instance is not None:
                try:
                    instance.cleanup()
                except Exception:
                    pass
        self.ocr_instance = None
        self.det_instance = None
        self.slide_instance = None


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    import uvicorn

    uvicorn.run("ddddocr.api:app", host=host, port=port, **kwargs)


__all__ = ["app", "create_app", "run_server", "DDDDOCRService"]
