"""GR-Net HTTP adapter. Missing/incompatible weights never trigger random inference."""
from __future__ import annotations
import base64
import hashlib
import importlib
import io
import os
from pathlib import Path
import threading
import time
import secrets

import numpy as np
from PIL import Image
import tifffile
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

MAX_UPLOAD = 50 * 1024 * 1024
MAX_PIXELS = 16_000_000


def read_image(data: bytes, name: str) -> np.ndarray:
    try:
        if name.lower().endswith((".tif", ".tiff")):
            with tifffile.TiffFile(io.BytesIO(data)) as tiff:
                shape = tiff.pages[0].shape
                if np.prod(shape) > MAX_PIXELS * 5:
                    raise ValueError("影像过大，请先切片。")
                array = tiff.pages[0].asarray()
            if array.ndim == 3 and array.shape[0] <= 5 and array.shape[-1] > 5:
                array = np.moveaxis(array, 0, -1)
        else:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("影像超过 1600 万像素。")
                array = np.asarray(image.convert("RGB") if image.mode in ("RGBA", "P", "LA") else image).copy()
        if array.ndim not in (2, 3) or array.shape[0] * array.shape[1] > MAX_PIXELS:
            raise ValueError("无效影像尺寸。")
        if not np.isfinite(array).all():
            raise ValueError("输入含 NaN / Inf，请先处理 NoData。")
        return array
    except (ValueError, OSError, tifffile.TiffFileError) as error:
        raise HTTPException(400, str(error)) from error


def preprocess(image: np.ndarray, ndvi: np.ndarray, band_order="bgrn", nir=None):
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise HTTPException(400, "模型需要三可见波段及一个近红外波段。")
    if image.dtype != np.uint8:
        raise HTTPException(400, "当前训练代码以 8 位影像 /255 输入；请按训练协议转换，不可直接送入 16 位 DN。")
    if image.shape[2] == 4:
        if band_order not in ("bgrn", "rgbn"):
            raise HTTPException(400, "band_order 应为 bgrn 或 rgbn。")
        bgrn = image if band_order == "bgrn" else image[:, :, [2, 1, 0, 3]]
    else:
        if nir is None or nir.ndim != 2 or nir.shape != image.shape[:2]:
            raise HTTPException(400, "RGB 输入需要独立、同尺寸的 NIR 文件。")
        if nir.dtype != np.uint8:
            raise HTTPException(400, "近红外波段需遵循训练协议，为 8 位数据。")
        # PNG/JPEG is visually RGB, independent of the TIFF band-order setting.
        bgrn = np.concatenate([image[:, :, [2, 1, 0]], nir[:, :, None]], axis=2)
    if ndvi.ndim == 3 and ndvi.shape[2] in (3, 4):
        if not np.array_equal(ndvi[:, :, 0], ndvi[:, :, 1]) or not np.array_equal(ndvi[:, :, 0], ndvi[:, :, 2]):
            raise HTTPException(400, "NDVI 必须为单波段，不能使用彩色可视化图片。")
        ndvi = ndvi[:, :, 0]
    if ndvi.ndim != 2 or ndvi.shape != image.shape[:2]:
        raise HTTPException(400, "NDVI 与原图尺寸不一致。")
    if ndvi.min() < -1 or ndvi.max() > 1:
        raise HTTPException(400, "NDVI 输入应为训练使用的 [-1,1] 数值或 0/1 编码。0/255 可视化掩膜不能直接替代。")
    # Exact normalization found in the user's data_loading.py/predict.py.
    return np.concatenate([bgrn.astype(np.float32) / 255.0,
                           ((ndvi.astype(np.float32) + 1.0) / 2.0)[:, :, None]], axis=2).transpose(2, 0, 1)


class ModelRunner:
    def __init__(self):
        self.model = None
        self.torch = None
        self.reason = "需要设置 GRNET_CHECKPOINT 为训练后的绿色屋顶专用权重。"
        self.checkpoint_sha256 = None
        self.device = "未加载"
        self.lock = threading.Lock()
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.loaded = True
        checkpoint = os.environ.get("GRNET_CHECKPOINT", "")
        if not checkpoint or not Path(checkpoint).is_file():
            return
        try:
            import torch
            self.torch = torch
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("权重不是有效 state_dict。")
            clean = {}
            for key, value in state.items():
                if key == "mask_values":
                    continue
                while key.startswith("module."):
                    key = key[7:]
                clean[key] = value
            if not any(key.startswith("additional_channels_conv.") for key in clean) or not any(key.startswith("last_layer_out_building.") for key in clean):
                raise ValueError("权重缺少绿顶专用五通道与建筑输出模块；VOC 预训练权重不能用于真实推理。")
            factory = os.environ.get("GRNET_FACTORY", "backend.model_code.hrnet:HRnet")
            module_name, class_name = factory.split(":", 1)
            module = importlib.import_module(module_name)
            # The supplied backbone contains its own CBAM implementation. The final
            # fusion uses the supplied identity fallback unless checkpoint keys
            # explicitly require the same implemented attention module.
            if module_name == "backend.model_code.hrnet" and any(k.startswith("cbam.") for k in clean):
                module.CBAM = importlib.import_module("backend.model_code.hrnet_backbone").CBAM
            model = getattr(module, class_name)(num_classes=1, backbone="hrnetv2_w32", pretrained=False,
                                               mask_thres=float(os.environ.get("GRNET_BUILDING_THRESHOLD", ".1")))
            model.load_state_dict(clean, strict=True)
            requested = os.environ.get("GRNET_DEVICE", "auto")
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
            model.to(self.device).eval()
            self.model = model
            self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            self.reason = None
        except Exception as error:
            self.model = None
            self.reason = f"模型未加载：{type(error).__name__}: {error}"

    def status(self):
        self.load()
        return {"ready": self.model is not None, "model": "用户 GR-Net 五通道双输出模型",
                "reason": self.reason, "device": self.device,
                "checkpoint_sha256": self.checkpoint_sha256}

    def predict(self, image):
        self.load()
        if self.model is None:
            raise HTTPException(503, self.reason)
        h, w = image.shape[1:]
        # Sliding tiles pad edges instead of resizing the input or discarding pixels.
        tile = int(os.environ.get("GRNET_TILE_SIZE", "300"))
        if not 64 <= tile <= 1024:
            raise HTTPException(500, "GRNET_TILE_SIZE 配置无效。")
        roof_probability = np.zeros((h, w), dtype=np.float32)
        green_probability = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        stride = max(32, tile // 2)
        ys = list(range(0, max(1, h - tile + 1), stride))
        xs = list(range(0, max(1, w - tile + 1), stride))
        if ys[-1] != max(0, h - tile): ys.append(max(0, h - tile))
        if xs[-1] != max(0, w - tile): xs.append(max(0, w - tile))
        with self.lock, self.torch.inference_mode():
            for y in ys:
                for x in xs:
                    patch = image[:, y:y+tile, x:x+tile]
                    ph, pw = patch.shape[1:]
                    patch = np.pad(patch, ((0, 0), (0, tile-ph), (0, tile-pw)), mode="edge")
                    tensor = self.torch.from_numpy(patch[None]).to(self.device)
                    outputs = self.model(tensor)
                    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                        raise HTTPException(500, "模型必须返回绿顶和建筑两类概率图。")
                    green, roof = [out[0, 0, :ph, :pw].detach().cpu().numpy() for out in outputs]
                    if not np.isfinite(green).all() or not np.isfinite(roof).all():
                        raise HTTPException(500, "模型返回无效数值。")
                    green_probability[y:y+ph, x:x+pw] += green
                    roof_probability[y:y+ph, x:x+pw] += roof
                    counts[y:y+ph, x:x+pw] += 1
        threshold = float(os.environ.get("GRNET_MASK_THRESHOLD", ".1"))
        roof_mask = roof_probability / counts >= threshold
        green_mask = (green_probability / counts >= threshold) & roof_mask
        return green_mask, roof_mask, threshold


runner = ModelRunner()
app = FastAPI(title="RoofScope GR-Net service", version="2.0.0")
origins = [s.strip() for s in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:4173").split(",") if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["Authorization", "Content-Type"], allow_credentials=False)


def authorize(authorization: str | None = Header(default=None)):
    token = os.environ.get("API_TOKEN", "")
    if token and not secrets.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "模型服务令牌无效。")


@app.get("/health", dependencies=[Depends(authorize)])
def health():
    return runner.status()


async def read_upload(file: UploadFile):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "单个输入超过 50 MB。")
    return read_image(data, file.filename or "image.png")


def png_base64(mask):
    buffer = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.post("/predict", dependencies=[Depends(authorize)])
async def predict(image: UploadFile = File(...), ndvi: UploadFile = File(...),
                  nir: UploadFile | None = File(default=None), band_order: str = Form(default="bgrn")):
    if not runner.status()["ready"]:
        raise HTTPException(503, runner.reason)
    data = preprocess(await read_upload(image), await read_upload(ndvi), band_order,
                      await read_upload(nir) if nir else None)
    started = time.perf_counter()
    green, roof, threshold = await run_in_threadpool(runner.predict, data)
    return {"green_mask": png_base64(green), "roof_mask": png_base64(roof),
            "threshold": threshold, "model": "GR-Net", "checkpoint_sha256": runner.checkpoint_sha256,
            "width": green.shape[1], "height": green.shape[0],
            "elapsed_ms": round((time.perf_counter() - started) * 1000)}
