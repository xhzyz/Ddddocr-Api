import numpy as np
from PIL import Image

from ddddocr.core.ocr_engine import OCREngine
from ddddocr.models.charset_manager import CharsetManager


def test_integer_charset_presets_keep_legacy_meaning():
    manager = CharsetManager(["", "0", "1", "a", "A", "中"])

    manager.set_ranges(0)
    assert manager.get_valid_indices() == [1, 2, 0]

    manager.set_ranges(7)
    assert manager.get_valid_indices() == [0, 5]


def test_restricted_inference_selects_best_allowed_class_and_probability_aliases():
    engine = OCREngine.__new__(OCREngine)
    engine.charset_manager = CharsetManager(["", "0", "1", "a", "A", "中"])
    engine.charset_manager.set_ranges(0)

    output = np.zeros((1, 2, 6), dtype=np.float32)
    output[0, 0, 3] = 10.0  # Globally best, but outside the digits preset.
    output[0, 0, 1] = 9.0
    output[0, 1, 0] = 10.0  # CTC blank.

    assert engine._process_text_output(output) == "0"
    probability = engine._process_probability_output(output)
    assert probability["text"] == "0"
    assert probability["charset"] == ["", "0", "1", "a", "A", "中"]
    assert probability["charsets"] == ["0", "1", ""]
    assert len(probability["probability"][0][0]) == 3


def test_custom_model_output_formats_and_normalization_are_supported():
    engine = OCREngine.__new__(OCREngine)
    engine.charset_manager = CharsetManager(["", "a", "b"])
    engine.use_import_onnx = True
    engine.word = False
    engine.resize = [8, 8]
    engine.channel = 3

    assert engine._process_custom_sequence_output(np.array([[1, 1, 0, 2]])) == "ab"
    assert engine._process_word_output([np.array([0]), np.array([1, 2])]) == "ab"

    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    normalized = engine._preprocess_image(image, png_fix=False)
    assert normalized.shape == (1, 3, 8, 8)
    assert normalized.dtype == np.float32
    assert float(normalized.max()) > 2.0


def test_per_call_charset_range_restores_persistent_range():
    engine = OCREngine.__new__(OCREngine)
    engine.charset_manager = CharsetManager(["", "0", "a"])
    engine.charset_manager.set_ranges("a")
    engine.is_ready = lambda: True
    engine._preprocess_image = lambda image, png_fix: np.zeros((1, 1, 1, 1), dtype=np.float32)
    engine._inference = lambda image, probability: engine.charset_manager.get_charset_range()

    during_call = engine.predict(Image.new("RGB", (1, 1)), charset_range="0")

    assert during_call == ["0", ""]
    assert engine.charset_manager.get_charset_range() == ["a", ""]
