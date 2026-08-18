# DdddOcr Docker API 部署

本目录提供的是完整 CPU/GPU HTTP API 封装，直接调用当前 `DdddOcr` SDK 的正确参数。
容器默认只绑定宿主机 `127.0.0.1:5555`，适合本机调用或放在 Nginx/Caddy 后面。

## 0. 功能完整性

这个 Docker API 没有删除 SDK 模型或算法，包含：

- 默认、旧版、Beta 和自定义 ONNX OCR；
- OCR 概率输出、PNG 透明背景修复、字符范围、颜色过滤；
- 目标检测；
- `slide_match` 与 `slide_comparison` 两种滑块算法；
- Base64、Data URI、文件上传和批量 OCR；
- CPU、NVIDIA GPU、API Key、CORS、实例缓存和 MCP 兼容入口；
- 旧 API 的 `/initialize`、`/switch-model`、`/toggle-feature`、`/detect`、`/status` 兼容路由。

CPU 镜像只是使用 `CPUExecutionProvider`，不是功能阉割。需要 CUDA 推理时使用 GPU Compose 覆盖文件。识别准确率仍由上游模型本身决定，Docker/API 不会提高或降低模型能力。

## 1. CPU 启动

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
docker compose logs -f ddddocr-api
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

检查服务：

- `GET http://127.0.0.1:5555/health`
- `GET http://127.0.0.1:5555/docs`

## 2. GPU 启动（NVIDIA）

先安装 NVIDIA 驱动和 NVIDIA Container Toolkit，然后执行：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

GPU 镜像使用 `onnxruntime-gpu` 和 CUDA 12 runtime。通过 `/health` 返回的
`providers` 字段可以确认是否出现 `CUDAExecutionProvider`。

## 3. OCR 调用

### 文件上传

```bash
curl -X POST http://127.0.0.1:5555/ocr/file \
  -F "file=@samples/yzm1.png"
```

### Base64 JSON

```python
import base64
import requests

with open("captcha.png", "rb") as file:
    image = base64.b64encode(file.read()).decode()

response = requests.post(
    "http://127.0.0.1:5555/ocr",
    json={
        "image": image,
        "probability": False,
        "png_fix": False,
        "colors": [],
        "custom_color_ranges": None,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["result"])
```

OCR 模型可通过查询参数选择：

```text
POST /ocr?beta=true
POST /ocr?old=true
POST /ocr?use_gpu=true&device_id=0
```

`old=true` 与 `beta=true` 不能同时使用。

## 4. 其他接口

| 接口 | 作用 |
|---|---|
| `POST /ocr`、`POST /ocr/file` | OCR 识别 |
| `POST /ocr/batch` | 批量 OCR，默认最多 32 张 |
| `POST /det`、`POST /det/file` | 目标检测 |
| `POST /slide_match` | 滑块模板匹配，返回坐标和 confidence |
| `POST /slide_comparison` | 两张图差异比较 |
| `POST /set_charset_range` | 设置字符范围；传 `null` 重置 |
| `GET /charset` | 获取字符集，`include_values=true` 返回内容 |
| `GET /model_info` | 查看实际加载的模型/执行提供程序 |
| `GET /instances` | 查看缓存的模型实例 |
| `POST /instances/cleanup?all=true` | 清理模型实例 |
| `GET /config` | 查看 API 配置和限制 |
| `GET /mcp/capabilities`、`POST /mcp/call` | MCP 工具协议兼容接口 |
| `POST /initialize`、`POST /switch-model` | 兼容旧版显式初始化/切换模型客户端 |
| `POST /toggle-feature`、`GET /status` | 兼容旧版功能开关和状态接口 |
| `POST /detect` | 旧版检测接口，需先调用 `/initialize`；新客户端直接使用 `/det` |

滑块请求示例：

```python
import base64
import requests

def b64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

response = requests.post(
    "http://127.0.0.1:5555/slide_match",
    json={
        "target_image": b64("target.png"),
        "background_image": b64("background.png"),
        "simple_target": True,
        "flag": False,  # 裁剪透明滑块失败时自动回退；True 表示直接返回错误
    },
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

`slide_match` 的 `result.target` 为 `[x1, y1, x2, y2]` 匹配框；
`target_x`、`target_y` 是透明滑块在原始小图中的裁剪偏移，`confidence` 为模板匹配置信度。

## 5. 自定义模型

`./custom_models` 会以只读方式挂载到容器 `/models`。例如：

```dotenv
DDDDOCR_IMPORT_ONNX_PATH=/models/my_model.onnx
DDDDOCR_CHARSETS_PATH=/models/charsets.json
```

两个路径必须同时设置。修改后执行：

```bash
docker compose up -d --build
```

## 6. API 鉴权和远程访问

在 `.env` 设置：

```dotenv
DDDDOCR_API_KEY=change-this-to-a-long-random-secret
DDDDOCR_BIND_IP=127.0.0.1
```

识别和管理接口支持 `X-API-Key` 或 `Authorization: Bearer ...`：

```bash
curl -H "X-API-Key: change-this-to-a-long-random-secret" \
  -X POST http://127.0.0.1:5555/ocr/file -F "file=@captcha.png"
```

`/health` 和 `/docs` 保持公开，便于健康检查和查看接口定义。公网部署建议使用
反向代理提供 HTTPS、限流和更严格的访问控制；不要直接把无鉴权的 8000 端口暴露到公网。

若确实需要其他机器直连，将 `DDDDOCR_BIND_IP` 改为 `0.0.0.0`，并在防火墙中只放行可信来源。

## 7. 运维命令

```bash
docker compose logs -f ddddocr-api
docker compose restart ddddocr-api
docker compose down
git pull
docker compose up -d --build
```

每个 worker 都会加载一份 ONNX 模型，建议先保持 `DDDDOCR_WORKERS=1`；需要提高吞吐时，
优先横向扩容容器，而不是盲目增加 worker 数量。
