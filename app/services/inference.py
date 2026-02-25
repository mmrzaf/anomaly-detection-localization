from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet34

logger = structlog.get_logger(__name__)

AllowedMethod = Literal["padim", "student"]


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def aggregate_image_score(
    score_map: np.ndarray,
    aggregation: str = "mean_topk",
    topk_frac: float = 0.1,
) -> float:
    s = np.asarray(score_map, dtype=np.float32).reshape(-1)
    if s.size == 0:
        return 0.0
    if aggregation == "max":
        return float(np.max(s))
    if aggregation == "mean":
        return float(np.mean(s))
    if aggregation == "mean_topk":
        k = max(1, int(round(float(topk_frac) * s.size)))
        return float(np.mean(np.partition(s, -k)[-k:]))
    raise ValueError(f"Unknown aggregation: {aggregation}")


def gaussian_smooth_np(m: np.ndarray, sigma: float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    radius = int(3 * float(sigma))
    if radius <= 0:
        return m.astype(np.float32, copy=False)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs ** 2) / (2.0 * float(sigma) ** 2))
    k = (k / np.sum(k)).astype(np.float32)

    tmp = np.pad(m, ((radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(m, dtype=np.float32)
    for i in range(m.shape[0]):
        out[i] = (tmp[i : i + 2 * radius + 1] * k[:, None]).sum(axis=0)

    tmp2 = np.pad(out, ((0, 0), (radius, radius)), mode="reflect")
    out2 = np.zeros_like(out, dtype=np.float32)
    for j in range(out.shape[1]):
        out2[:, j] = (tmp2[:, j : j + 2 * radius + 1] * k[None, :]).sum(axis=1)
    return out2.astype(np.float32)


def sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
    ):
        super().__init__()

        def _load_resnet18(pretrained_flag: bool):
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained_flag else None
                return resnet18(weights=weights)
            except Exception:
                return resnet18(pretrained=pretrained_flag)

        def _load_resnet34(pretrained_flag: bool):
            try:
                weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained_flag else None
                return resnet34(weights=weights)
            except Exception:
                return resnet34(pretrained=pretrained_flag)

        if backbone == "resnet18":
            self.backbone = _load_resnet18(pretrained)
        elif backbone == "resnet34":
            self.backbone = _load_resnet34(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.layers = tuple(layers)
        valid = {"layer1", "layer2", "layer3", "layer4"}
        bad = [x for x in self.layers if x not in valid]
        if bad:
            raise ValueError(f"Unsupported layers: {bad}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: dict[str, torch.Tensor] = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        feats["layer1"] = x
        x = self.backbone.layer2(x)
        feats["layer2"] = x
        x = self.backbone.layer3(x)
        feats["layer3"] = x
        x = self.backbone.layer4(x)
        feats["layer4"] = x
        return [feats[k] for k in self.layers]


def align_and_concat_features(
    feature_list: list[torch.Tensor],
    target_hw: tuple[int, int] = (28, 28),
) -> torch.Tensor:
    aligned: list[torch.Tensor] = []
    for f in feature_list:
        if tuple(f.shape[-2:]) != tuple(target_hw):
            f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
        aligned.append(f)
    return torch.cat(aligned, dim=1)


@dataclass(frozen=True)
class LoadedArtifact:
    key: str
    file_path: Path
    method: AllowedMethod
    category: str
    backbone: str
    pretrained: bool
    layers: tuple[str, ...]
    target_hw: tuple[int, int]
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    d: int
    smooth_sigma: float | None
    image_aggregation: str
    topk_fraction: float
    train_score_mean: float
    train_score_std: float
    threshold_raw: float
    threshold_quantile: float
    threshold_z: float
    channel_index: np.ndarray
    mu_map: torch.Tensor
    cov_inv: torch.Tensor | None
    std_map: torch.Tensor | None
    schema_version: str
    created_at: str | None
    export_source: str | None
    artifact_version: str | None


class InferenceService:
    def __init__(self, artifacts_dir: str | Path, device: str | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device or _auto_device()
        self._artifacts: dict[str, LoadedArtifact] = {}
        self._feature_models: dict[tuple[str, bool, tuple[str, ...]], ResNetFeatureExtractor] = {}
        self._model_cache_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        manifest_path = self.artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Artifact manifest not found: {manifest_path}. Export artifacts from notebook first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError(f"Manifest has no models: {manifest_path}")

        loaded: dict[str, LoadedArtifact] = {}
        for item in models:
            rel_path = item["path"]
            artifact_path = (self.artifacts_dir / rel_path).resolve()
            payload = torch.load(artifact_path, map_location="cpu")
            method = str(payload["method"]).lower()
            category = str(payload["category"])
            if method not in {"padim", "student"}:
                raise ValueError(f"Unsupported method in artifact {artifact_path}: {method}")
            key = self._artifact_key(method, category)

            channel_index_t = payload["channel_index"]
            if torch.is_tensor(channel_index_t):
                channel_index = channel_index_t.detach().cpu().numpy().astype(np.int64)
            else:
                channel_index = np.asarray(channel_index_t, dtype=np.int64)

            mu_map = payload["mu_map"].detach().cpu().float()
            cov_inv = payload.get("cov_inv")
            if cov_inv is not None:
                cov_inv = cov_inv.detach().cpu().float()
            std_map = payload.get("std_map")
            if std_map is not None:
                std_map = std_map.detach().cpu().float()

            loaded[key] = LoadedArtifact(
                key=key,
                file_path=artifact_path,
                method=method,  # type: ignore[arg-type]
                category=category,
                backbone=str(payload["backbone"]),
                pretrained=bool(payload.get("pretrained", True)),
                layers=tuple(payload["layers"]),
                target_hw=tuple(int(x) for x in payload["target_hw"]),
                input_size=tuple(int(x) for x in payload.get("input_size", [224, 224])),
                mean=tuple(float(x) for x in payload.get("mean", [0.485, 0.456, 0.406])),
                std=tuple(float(x) for x in payload.get("std", [0.229, 0.224, 0.225])),
                d=int(payload.get("d", int(channel_index.size))),
                smooth_sigma=(None if payload.get("smooth_sigma") is None else float(payload["smooth_sigma"])),
                image_aggregation=str(payload.get("image_aggregation", "mean_topk")),
                topk_fraction=float(payload.get("topk_fraction", 0.1)),
                train_score_mean=float(payload.get("train_score_mean", 0.0)),
                train_score_std=max(float(payload.get("train_score_std", 1.0)), 1e-12),
                threshold_raw=float(payload["threshold_raw"]),
                threshold_quantile=float(payload.get("threshold_quantile", 0.995)),
                threshold_z=float(payload.get("threshold_z", 0.0)),
                channel_index=channel_index,
                mu_map=mu_map,
                cov_inv=cov_inv,
                std_map=std_map,
                schema_version=str(payload.get("schema_version", "unknown")),
                created_at=payload.get("created_at"),
                export_source=payload.get("export_source"),
                artifact_version=payload.get("artifact_version"),
            )

        self._artifacts = loaded
        self._loaded = True
        logger.info(
            "Loaded inference artifacts",
            artifact_dir=str(self.artifacts_dir),
            count=len(self._artifacts),
            device=self.device,
        )

    def close(self) -> None:
        self._feature_models.clear()
        self._artifacts.clear()
        self._loaded = False

    def list_models(self) -> list[dict[str, Any]]:
        items = []
        for art in sorted(self._artifacts.values(), key=lambda a: (a.category, a.method)):
            items.append(
                {
                    "method": art.method,
                    "category": art.category,
                    "artifact_version": art.artifact_version,
                    "schema_version": art.schema_version,
                    "backbone": art.backbone,
                    "layers": list(art.layers),
                    "target_hw": list(art.target_hw),
                    "threshold_raw": art.threshold_raw,
                    "threshold_quantile": art.threshold_quantile,
                    "created_at": art.created_at,
                }
            )
        return items

    def supported_categories(self) -> list[str]:
        return sorted({a.category for a in self._artifacts.values()})

    def has_model(self, method: str, category: str) -> bool:
        return self._artifact_key(method, category) in self._artifacts

    def predict(
        self,
        image_bytes: bytes,
        method: AllowedMethod,
        category: str,
        return_visuals: bool = False,
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Inference service not loaded")
        art = self._get_artifact(method, category)

        t0 = time.perf_counter()
        pil_image = self._decode_image(image_bytes)
        orig_w, orig_h = pil_image.size

        x = self._preprocess(pil_image, art).to(self.device)
        t_pre = time.perf_counter()

        with self._infer_lock, torch.no_grad():
            feature_model = self._get_feature_model(art)
            feats = feature_model(x)
            emb = align_and_concat_features(feats, target_hw=art.target_hw)
            ch_idx = torch.as_tensor(art.channel_index, dtype=torch.long, device=self.device)
            emb = emb.index_select(dim=1, index=ch_idx)

            if art.method == "padim":
                score_map = self._padim_score_map(emb, art)
            else:
                score_map = self._student_score_map(emb, art)

        score_map = score_map.astype(np.float32)
        if art.smooth_sigma is not None and float(art.smooth_sigma) > 0:
            score_map = gaussian_smooth_np(score_map, float(art.smooth_sigma))

        raw_score = aggregate_image_score(
            score_map,
            aggregation=art.image_aggregation,
            topk_frac=art.topk_fraction,
        )
        z_score = (float(raw_score) - art.train_score_mean) / max(art.train_score_std, 1e-12)
        calibrated_score = sigmoid(z_score)
        threshold_calibrated = sigmoid(float(art.threshold_z))
        is_anomalous = bool(raw_score >= art.threshold_raw)
        t_inf = time.perf_counter()

        visuals: dict[str, str | None] = {
            "original_image_base64": None,
            "heatmap_image_base64": None,
            "overlay_image_base64": None,
        }
        visual_content_type = "image/png"
        if return_visuals:
            visuals = self._render_visuals(
                original=pil_image,
                score_map=score_map,
            )
            visual_content_type = "image/png"
        t_vis = time.perf_counter()

        return {
            "method": art.method,
            "category": art.category,
            "image_width": orig_w,
            "image_height": orig_h,
            "score_map_height": int(score_map.shape[0]),
            "score_map_width": int(score_map.shape[1]),
            "raw_image_score": float(raw_score),
            "score_z": float(z_score),
            "calibrated_score": float(calibrated_score),
            "threshold_raw": float(art.threshold_raw),
            "threshold_calibrated": float(threshold_calibrated),
            "threshold_quantile": float(art.threshold_quantile),
            "is_anomalous": is_anomalous,
            "model_meta": {
                "artifact_version": art.artifact_version,
                "schema_version": art.schema_version,
                "backbone": art.backbone,
                "layers": list(art.layers),
                "target_hw": list(art.target_hw),
                "export_source": art.export_source,
                "created_at": art.created_at,
            },
            "visual_content_type": visual_content_type if return_visuals else None,
            "timings_ms": {
                "total": round((t_vis - t0) * 1000.0, 2),
                "preprocess": round((t_pre - t0) * 1000.0, 2),
                "inference_and_scoring": round((t_inf - t_pre) * 1000.0, 2),
                "render_visuals": round((t_vis - t_inf) * 1000.0, 2),
            },
            **visuals,
        }

    def _artifact_key(self, method: str, category: str) -> str:
        return f"{method.lower()}::{category.lower()}"

    def _get_artifact(self, method: str, category: str) -> LoadedArtifact:
        key = self._artifact_key(method, category)
        art = self._artifacts.get(key)
        if art is None:
            available = sorted((a.method, a.category) for a in self._artifacts.values())
            raise ValueError(
                f"Model artifact not found for method='{method}', category='{category}'. "
                f"Available pairs: {available}"
            )
        return art

    def _get_feature_model(self, art: LoadedArtifact) -> ResNetFeatureExtractor:
        key = (art.backbone, art.pretrained, art.layers)
        model = self._feature_models.get(key)
        if model is not None:
            return model
        with self._model_cache_lock:
            model = self._feature_models.get(key)
            if model is None:
                model = ResNetFeatureExtractor(
                    backbone=art.backbone,
                    pretrained=art.pretrained,
                    layers=art.layers,
                ).to(self.device)
                model.eval()
                self._feature_models[key] = model
                logger.info(
                    "Initialized feature backbone",
                    backbone=art.backbone,
                    layers=list(art.layers),
                    device=self.device,
                )
        return model

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise ValueError("Uploaded image is empty")
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.load()
            return img
        except Exception as e:
            raise ValueError("Uploaded file is not a valid image") from e

    def _preprocess(self, image: Image.Image, art: LoadedArtifact) -> torch.Tensor:
        tfm = transforms.Compose(
            [
                transforms.Resize(tuple(art.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(art.mean), std=list(art.std)),
            ]
        )
        x = tfm(image).unsqueeze(0)
        return x

    def _padim_score_map(self, emb: torch.Tensor, art: LoadedArtifact) -> np.ndarray:
        if art.cov_inv is None:
            raise RuntimeError("PaDiM artifact missing cov_inv")
        x = emb.float()
        n, c, h, w = x.shape
        if n != 1:
            raise RuntimeError(f"Expected batch size 1 for API inference, got {n}")
        mu = art.mu_map.to(self.device).float()
        inv = art.cov_inv.to(self.device).float()

        if tuple(mu.shape) != (c, h, w):
            raise RuntimeError(f"mu_map shape mismatch: expected {(c,h,w)}, got {tuple(mu.shape)}")
        if tuple(inv.shape) != (h * w, c, c):
            raise RuntimeError(f"cov_inv shape mismatch: expected {(h*w,c,c)}, got {tuple(inv.shape)}")

        x_lc = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
        mu_lc = mu.permute(1, 2, 0).reshape(h * w, c)
        delta = x_lc - mu_lc.unsqueeze(0)
        d2 = torch.einsum("nlc,lcd,nld->nl", delta, inv, delta)
        d2 = torch.clamp(d2, min=0.0)
        scores = torch.sqrt(d2 + 1e-12)
        return scores.reshape(h, w).detach().cpu().numpy().astype(np.float32)

    def _student_score_map(self, emb: torch.Tensor, art: LoadedArtifact) -> np.ndarray:
        if art.std_map is None:
            raise RuntimeError("Student artifact missing std_map")
        x = emb.float()
        n, c, h, w = x.shape
        if n != 1:
            raise RuntimeError(f"Expected batch size 1 for API inference, got {n}")
        mu = art.mu_map.to(self.device).float()
        std = art.std_map.to(self.device).float().clamp_min(1e-6)

        if tuple(mu.shape) != (c, h, w):
            raise RuntimeError(f"mu_map shape mismatch: expected {(c,h,w)}, got {tuple(mu.shape)}")
        if tuple(std.shape) != (c, h, w):
            raise RuntimeError(f"std_map shape mismatch: expected {(c,h,w)}, got {tuple(std.shape)}")

        z = (x - mu.unsqueeze(0)) / std.unsqueeze(0)
        maps = torch.sqrt(torch.mean(z * z, dim=1) + 1e-6)
        return maps.reshape(h, w).detach().cpu().numpy().astype(np.float32)

    def _render_visuals(
        self,
        original: Image.Image,
        score_map: np.ndarray,
    ) -> dict[str, str]:
        orig_rgb = original.convert("RGB")
        w, h = orig_rgb.size
        norm = normalize_01(score_map)
        sm_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").resize((w, h), resample=Image.BILINEAR)
        sm = np.asarray(sm_img, dtype=np.float32) / 255.0

        heat = np.zeros((h, w, 3), dtype=np.uint8)
        # Simple red-yellow heatmap (no extra deps)
        heat[..., 0] = np.clip(255.0 * sm, 0, 255).astype(np.uint8)  # red
        heat[..., 1] = np.clip(255.0 * np.power(sm, 1.8), 0, 255).astype(np.uint8)  # yellow at highs
        heat[..., 2] = np.clip(80.0 * np.power(1.0 - sm, 2.0), 0, 255).astype(np.uint8)

        orig_np = np.asarray(orig_rgb, dtype=np.uint8)
        alpha = (0.15 + 0.55 * sm)[..., None].astype(np.float32)
        overlay = np.clip((1.0 - alpha) * orig_np.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)

        original_png_b64 = self._to_png_b64(orig_rgb)
        heatmap_png_b64 = self._to_png_b64(Image.fromarray(heat, mode="RGB"))
        overlay_png_b64 = self._to_png_b64(Image.fromarray(overlay, mode="RGB"))

        return {
            "original_image_base64": original_png_b64,
            "heatmap_image_base64": heatmap_png_b64,
            "overlay_image_base64": overlay_png_b64,
        }

    def _to_png_b64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
