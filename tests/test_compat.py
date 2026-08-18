import io

from PIL import Image

from ddddocr.compat.v1 import DdddOcr
from ddddocr.core.slide_engine import SlideEngine


class FakeOCREngine:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"


class FakeDetectionEngine:
    def __init__(self):
        self.calls = []

    def predict(self, value):
        self.calls.append(value)
        return []


class FakeSlideEngine:
    def __init__(self):
        self.calls = []

    def slide_match(self, *args, **kwargs):
        self.calls.append(("match", args, kwargs))
        return {}

    def slide_comparison(self, *args):
        self.calls.append(("comparison", args))
        return {}


def test_legacy_classification_aliases_and_charset_range():
    instance = DdddOcr.__new__(DdddOcr)
    instance.det = False
    instance.ocr_engine = FakeOCREngine()

    assert instance.classification(
        b"image",
        colors=["red"],
        custom_color_ranges=[((0, 0, 0), (1, 1, 1))],
        charset_range="0123",
    ) == "ok"
    call = instance.ocr_engine.calls[0]
    assert call["color_filter_colors"] == ["red"]
    assert call["color_filter_custom_ranges"] == [((0, 0, 0), (1, 1, 1))]
    assert call["charset_range"] == "0123"


def test_named_custom_color_ranges_are_flattened():
    instance = DdddOcr.__new__(DdddOcr)
    instance.det = False
    instance.ocr_engine = FakeOCREngine()
    instance.classification(
        b"image",
        colors=["light_blue"],
        custom_color_ranges={"light_blue": [(90, 30, 30), (110, 255, 255)]},
    )
    call = instance.ocr_engine.calls[0]
    assert call["color_filter_colors"] is None
    assert call["color_filter_custom_ranges"] == [((90, 30, 30), (110, 255, 255))]


def test_legacy_keyword_names_for_detection_and_sliders():
    instance = DdddOcr.__new__(DdddOcr)
    instance.det = True
    instance.detection_engine = FakeDetectionEngine()
    instance.slide_engine = FakeSlideEngine()

    instance.detection(img_bytes=b"image")
    instance.slide_match(
        target_bytes=b"target",
        background_bytes=b"background",
        flag=True,
    )
    instance.slide_comparison(
        target_bytes=b"target",
        background_bytes=b"background",
        flag=True,
    )

    assert instance.detection_engine.calls == [b"image"]
    assert instance.slide_engine.calls[0][0] == "match"
    assert instance.slide_engine.calls[1][0] == "comparison"


def test_legacy_detection_base64_keyword_is_accepted():
    instance = DdddOcr.__new__(DdddOcr)
    instance.det = True
    instance.detection_engine = FakeDetectionEngine()

    instance.detection(img_base64="aGVsbG8=")

    assert instance.detection_engine.calls == ["aGVsbG8="]


def test_legacy_get_target_and_helper_methods_are_exposed():
    instance = DdddOcr.__new__(DdddOcr)
    instance.slide_engine = SlideEngine()
    instance.detection_engine = None

    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.paste((0, 0, 0, 255), (3, 4, 7, 8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    cropped, x, y = instance.get_target(target_bytes=buffer.getvalue())

    assert cropped.size == (4, 4)
    assert (x, y) == (3, 4)
    assert callable(instance.preproc)
    assert callable(instance.nms)
