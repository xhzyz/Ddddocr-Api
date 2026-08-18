import base64
import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


api_module = importlib.import_module("ddddocr.api.app")


class FakeDdddOcr:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.charset_range = None
        self.cleaned = False
        self.__class__.instances.append(self)

    def classification(self, image, **kwargs):
        self.calls.append(("classification", image, kwargs))
        self.charset_range = kwargs.get("charset_range")
        if kwargs.get("probability"):
            return {"charsets": ["a"], "probability": [0.99]}
        return "fake-ocr"

    def detection(self, image):
        self.calls.append(("detection", image))
        return [[1, 2, 3, 4]]

    def slide_match(self, target, background, simple_target=False, flag=False):
        self.calls.append(("slide_match", target, background, simple_target, flag))
        return {
            "target": [12, 34, 32, 54],
            "target_x": 0,
            "target_y": 0,
            "confidence": 0.98,
        }

    def slide_comparison(self, target, background):
        self.calls.append(("slide_comparison", target, background))
        return {"target": [5, 6], "target_x": 5, "target_y": 6}

    def set_ranges(self, charset_range):
        self.charset_range = charset_range

    def get_charset(self):
        return ["", "0", "1", "a"]

    def get_model_info(self):
        return {
            "ocr_enabled": self.init_kwargs.get("ocr", False),
            "det_enabled": self.init_kwargs.get("det", False),
            "providers": ["CPUExecutionProvider"],
        }

    def cleanup(self):
        self.cleaned = True


def image_bytes(fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), color=(255, 255, 255)).save(buffer, format=fmt)
    return buffer.getvalue()


def image_base64():
    return base64.b64encode(image_bytes()).decode()


def find_instance(*, call_name=None, **init_kwargs):
    for instance in FakeDdddOcr.instances:
        if any(instance.init_kwargs.get(key) != value for key, value in init_kwargs.items()):
            continue
        if call_name is not None and not any(call[0] == call_name for call in instance.calls):
            continue
        return instance
    raise AssertionError("matching fake DdddOcr instance not found")


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    api_module.registry.cleanup_all()
    api_module.legacy_runtime.reset()
    FakeDdddOcr.instances.clear()
    monkeypatch.setattr(api_module, "DdddOcr", FakeDdddOcr)
    monkeypatch.setattr(api_module, "API_KEY", "")
    yield
    api_module.legacy_runtime.reset()
    api_module.registry.cleanup_all()


@pytest.fixture
def client():
    with TestClient(api_module.app) as value:
        yield value


def test_ocr_uses_current_sdk_keywords(client):
    response = client.post(
        "/ocr?beta=true",
        json={
            "image": image_base64(),
            "png_fix": True,
            "probability": False,
            "colors": ["red"],
            "custom_color_ranges": [[[0, 50, 50], [10, 255, 255]]],
            "charset_range": "0123456789",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"] == "fake-ocr"
    engine = find_instance(beta=True, call_name="classification")
    assert engine.init_kwargs["beta"] is True
    assert engine.charset_range == "0123456789"
    name, data, kwargs = engine.calls[0]
    assert name == "classification"
    assert data.startswith(b"\x89PNG")
    assert kwargs == {
        "png_fix": True,
        "probability": False,
        "color_filter_colors": ["red"],
        "color_filter_custom_ranges": [((0, 50, 50), (10, 255, 255))],
        "charset_range": "0123456789",
    }


def test_custom_named_color_is_supported(client):
    response = client.post(
        "/ocr",
        json={
            "image": image_base64(),
            "colors": ["light_blue"],
            "custom_color_ranges": {
                "light_blue": [[90, 30, 30], [110, 255, 255]],
            },
        },
    )

    assert response.status_code == 200, response.text
    _, _, kwargs = FakeDdddOcr.instances[0].calls[0]
    assert kwargs["color_filter_colors"] is None
    assert kwargs["color_filter_custom_ranges"] == [((90, 30, 30), (110, 255, 255))]


def test_ocr_file_and_probability(client):
    response = client.post(
        "/ocr/file",
        files={"file": ("captcha.png", image_bytes(), "image/png")},
        data={"probability": "true", "png_fix": "true"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["charsets"] == ["a"]
    _, _, kwargs = FakeDdddOcr.instances[0].calls[0]
    assert kwargs["probability"] is True
    assert kwargs["png_fix"] is True


def test_detection_passes_image_positionally(client):
    response = client.post("/det", json={"image": image_base64()})

    assert response.status_code == 200, response.text
    assert response.json()["result"] == [[1, 2, 3, 4]]
    assert find_instance(call_name="detection").calls[0][0] == "detection"


def test_slider_responses_keep_confidence_and_integer_coordinates(client):
    payload = {
        "target_image": image_base64(),
        "background_image": image_base64(),
        "simple_target": True,
        "flag": True,
    }
    match = client.post("/slide_match", json=payload)
    comparison = client.post(
        "/slide_comparison",
        json={
            "target_image": image_base64(),
            "background_image": image_base64(),
        },
    )

    assert match.status_code == 200, match.text
    assert match.json()["result"]["confidence"] == 0.98
    assert comparison.status_code == 200, comparison.text
    assert comparison.json()["result"]["target_x"] == 5


def test_charset_management_and_reset(client):
    update = client.post("/set_charset_range", json={"charset_range": ["0", "1"]})
    charset = client.get("/charset?include_values=true")
    reset = client.post("/set_charset_range", json={"charset_range": None})

    assert update.status_code == 200, update.text
    assert charset.status_code == 200, charset.text
    assert charset.json()["charset"] == ["", "0", "1", "a"]
    assert reset.status_code == 200, reset.text
    assert reset.json()["charset_range"] is None


def test_batch_reports_item_errors_without_failing_whole_request(client):
    response = client.post(
        "/ocr/batch",
        json={
            "images": [
                {"image": image_base64()},
                {"image": "not-base64"},
            ]
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"][0]["result"] == "fake-ocr"
    assert "Base64" in body["results"][1]["error"]


def test_optional_api_key_protects_non_health_endpoints(client, monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "test-secret")

    assert client.get("/health").status_code == 200
    assert client.post("/ocr", json={"image": image_base64()}).status_code == 401
    authorized = client.post(
        "/ocr",
        json={"image": image_base64()},
        headers={"X-API-Key": "test-secret"},
    )
    assert authorized.status_code == 200, authorized.text


def test_rejects_invalid_color_and_oversized_dimensions(client, monkeypatch):
    invalid_color = client.post(
        "/ocr",
        json={"image": image_base64(), "colors": ["magenta"]},
    )
    assert invalid_color.status_code == 400

    monkeypatch.setattr(api_module, "MAX_IMAGE_SIDE", 5)
    oversized = client.post("/ocr", json={"image": image_base64()})
    assert oversized.status_code == 413


def test_mcp_facade_uses_same_api_implementation(client):
    capabilities = client.get("/mcp/capabilities")
    call = client.post(
        "/mcp/call",
        json={
            "method": "ddddocr_ocr",
            "params": {"image": image_base64(), "probability": False},
            "id": 7,
        },
    )

    assert capabilities.status_code == 200
    assert any(tool["name"] == "ddddocr_ocr" for tool in capabilities.json()["tools"])
    assert call.status_code == 200, call.text
    assert call.json()["id"] == 7
    assert call.json()["result"] == "fake-ocr"


def test_legacy_management_and_detection_endpoints_remain_available(client):
    initialized = client.post("/initialize", json={"ocr": True, "det": True})
    status = client.get("/status")
    detected = client.post("/detect", json={"image": image_base64()})

    assert initialized.status_code == 200, initialized.text
    assert initialized.json()["success"] is True
    assert status.status_code == 200, status.text
    assert set(status.json()["loaded_models"]) == {"ocr", "detection", "slide"}
    assert detected.status_code == 200, detected.text
    assert detected.json()["data"]["bboxes"] == [[1, 2, 3, 4]]


def test_legacy_ocr_color_field_names_are_accepted(client):
    response = client.post(
        "/ocr",
        json={
            "image": image_base64(),
            "color_filter_colors": ["red"],
            "color_filter_custom_ranges": [[[0, 50, 50], [10, 255, 255]]],
        },
    )

    assert response.status_code == 200, response.text
    _, _, kwargs = FakeDdddOcr.instances[0].calls[0]
    assert kwargs["color_filter_colors"] == ["red"]
    assert kwargs["color_filter_custom_ranges"] == [((0, 50, 50), (10, 255, 255))]
