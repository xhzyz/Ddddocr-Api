# coding=utf-8
"""Canonical DdddOcr HTTP API exports."""

from .app import app, create_app, main
from .server import DDDDOCRService
# Keep the request/response model imports that older integrations obtained
# from ``ddddocr.api``.
from .models import (
    APIResponse,
    DetectionRequest,
    DetectionResponse,
    InitializeRequest,
    MCPCapabilities,
    MCPRequest,
    MCPResponse,
    OCRRequest,
    OCRResponse,
    SlideComparisonRequest,
    SlideMatchRequest,
    SlideResponse,
    StatusResponse,
    SwitchModelRequest,
    ToggleFeatureRequest,
)

__version__ = "1.6.1"
__author__ = "sml2h3"


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Backward-compatible programmatic server entry point."""
    import uvicorn

    uvicorn.run("ddddocr.api:app", host=host, port=port, **kwargs)


__all__ = [
    "app",
    "create_app",
    "main",
    "run_server",
    "DDDDOCRService",
    "APIResponse",
    "DetectionRequest",
    "DetectionResponse",
    "InitializeRequest",
    "MCPCapabilities",
    "MCPRequest",
    "MCPResponse",
    "OCRRequest",
    "OCRResponse",
    "SlideComparisonRequest",
    "SlideMatchRequest",
    "SlideResponse",
    "StatusResponse",
    "SwitchModelRequest",
    "ToggleFeatureRequest",
]
