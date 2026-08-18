#!/bin/sh
set -eu

exec python -m ddddocr api \
  --host="${DDDDOCR_HOST:-0.0.0.0}" \
  --port="${DDDDOCR_PORT:-8000}" \
  --workers="${DDDDOCR_WORKERS:-1}" \
  --ocr="${DDDDOCR_OCR:-true}" \
  --det="${DDDDOCR_DET:-false}" \
  --old="${DDDDOCR_OLD:-false}" \
  --beta="${DDDDOCR_BETA:-false}" \
  --use-gpu="${DDDDOCR_USE_GPU:-false}" \
  --device-id="${DDDDOCR_DEVICE_ID:-0}" \
  --show-ad="${DDDDOCR_SHOW_AD:-false}" \
  --import-onnx-path="${DDDDOCR_IMPORT_ONNX_PATH:-}" \
  --charsets-path="${DDDDOCR_CHARSETS_PATH:-}"
