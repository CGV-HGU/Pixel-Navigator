# cv_utils/yoloe_tools.py
# YOLOE segmentation-only utilities (no bbox usage)

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

# ← 여기! 프로젝트의 기본 체크포인트 경로를 constants에서 가져온다.
from constants import YOLOE_CHECKPOINT_PATH


@dataclass
class Detections:
    """
    Segmentation-only results to plug into your planner.
    - masks: (N, H, W) uint8 or bool
    - class_id: (N,) int indices aligned to `class_names`
    - confidence: (N,) float (optional; derived from model boxes)
    - class_names: list of class names used for prompting
    """
    masks: Optional[np.ndarray]          # (N,H,W) {0,1}
    class_id: np.ndarray                 # (N,)
    confidence: np.ndarray               # (N,)
    class_names: List[str]


def _project_root() -> Path:
    # .../Pixel-Navigator/cv_utils/yoloe_tools.py → project root
    return Path(__file__).resolve().parents[1]


def _resolve_weights_path(weights: Optional[str]) -> str:
    """
    Resolve YOLOE weights path with these fallbacks (first hit wins):
      1) explicit `weights` arg (as-is, then relative to project root)
      2) env var YOLOE_WEIGHTS
      3) constants.YOLOE_CHECKPOINT_PATH (as-is, then relative to project root)
    """
    def _try_paths(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        p = Path(raw)
        if p.is_file():
            return str(p)
        pr = _project_root() / raw.lstrip("./")
        if pr.is_file():
            return str(pr)
        return None

    # 1) explicit arg
    hit = _try_paths(weights)
    if hit:
        return hit

    # 2) env override
    hit = _try_paths(os.environ.get("YOLOE_WEIGHTS"))
    if hit:
        return hit

    # 3) constants default
    hit = _try_paths(YOLOE_CHECKPOINT_PATH)
    if hit:
        return hit

    # Build hint list
    candidates = []
    for base in [(_project_root() / "checkpoints"), (Path.cwd() / "checkpoints")]:
        if base.is_dir():
            candidates += [str(x) for x in base.glob("yoloe*-seg.pt")]
    hint = "\nAvailable in checkpoints:\n  " + "\n  ".join(candidates) if candidates else ""
    raise FileNotFoundError(
        "YOLOE weights not found.\n"
        f"Tried:\n"
        f"  weights arg: {weights}\n"
        f"  env YOLOE_WEIGHTS: {os.environ.get('YOLOE_WEIGHTS')}\n"
        f"  constants.YOLOE_CHECKPOINT_PATH: {YOLOE_CHECKPOINT_PATH}\n"
        + hint
    )


def initialize_yoloe_model(
    weights: Optional[str] = None,       # ← 기본은 None → constants 경로 사용
    device: str = "cuda:0",
    classes: Optional[List[str]] = None,
    prompt_mode: str = "text",           # "text" | "prompt_free"
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> YOLOE:
    """
    Initialize YOLOE (seg weights required). Optionally set text-prompt classes.

    - 기본 가중치: constants.YOLOE_CHECKPOINT_PATH (예: './checkpoints/yoloe-11s-seg.pt')
    - 다른 ckpt로 바꾸려면:
        initialize_yoloe_model(weights="./checkpoints/yoloe-11l-seg.pt")
      또는 환경변수:
        export YOLOE_WEIGHTS=/abs/path/to/yoloe-11s-seg.pt
    """
    weight_path = _resolve_weights_path(weights)
    model = YOLOE(weight_path)
    try:
        model.to(device)
    except Exception:
        pass

    # Default inference overrides
    model.overrides = getattr(model, "overrides", {})
    model.overrides["conf"] = conf_threshold
    model.overrides["iou"] = iou_threshold

    if classes and prompt_mode == "text":
        set_yoloe_classes(model, classes)
    return model

def set_yoloe_classes(model: YOLOE, classes: List[str]) -> List[str]:
    """Configure YOLOE text-prompt classes with simple LRU-like cache."""
    # 정규화된 키 (순서 포함)
    key = tuple([str(c).strip().lower() for c in classes])

    # 모델 객체에 캐시 dict 달아둠
    cache = getattr(model, "_text_pe_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_text_pe_cache", cache)

    if key in cache:
        pe = cache[key]
    else:
        pe = model.get_text_pe(classes)   # <-- 비용 큰 호출
        # 캐시 크기 제한 (예: 16)
        if len(cache) >= 16:
            cache.clear()
        cache[key] = pe

    model.set_classes(classes, pe)
    return classes


def _ensure_hw(image: np.ndarray) -> Tuple[int, int]:
    h, w = (image.shape[0], image.shape[1]) if image.ndim >= 2 else (480, 640)
    return int(h), int(w)


def yoloe_detection(
    image: np.ndarray,
    target_classes: List[str],
    model: YOLOE,
    box_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    run_extra_nms: bool = False,     # kept for API compatibility; unused (seg-only)
    use_text_prompt: bool = True,
    retina_masks: bool = True,
) -> Detections:
    """
    Single-pass YOLOE inference returning ONLY segmentation masks and class ids.
    - If `use_text_prompt=True`, sets model classes to `target_classes` each call.
    - Returns empty arrays if nothing found.
    """
    if use_text_prompt:
        set_yoloe_classes(model, target_classes)

    results = model.predict(
        image,
        conf=box_threshold,
        iou=iou_threshold,
        retina_masks=retina_masks,
        verbose=False,
    )
    r = results[0]
    H, W = _ensure_hw(image)

    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return Detections(
            masks=None,
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=float),
            class_names=target_classes,
        )

    cls_raw = r.boxes.cls.detach().cpu().numpy().astype(int)      # (N,)
    conf = r.boxes.conf.detach().cpu().numpy().astype(float)      # (N,)

    names_map = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names or [])}
    inv = {name.lower(): i for i, name in enumerate(target_classes)}
    mapped = []
    for c in cls_raw:
        name = str(names_map.get(int(c), "")).lower()
        mapped.append(inv.get(name, -1))
    mapped = np.array(mapped, dtype=int)

    keep = mapped >= 0
    conf = conf[keep]
    mapped = mapped[keep]

    masks = None
    if getattr(r, "masks", None) is not None and r.masks is not None and len(r.masks) > 0:
        m = r.masks.data.detach().cpu().numpy().astype(np.uint8)  # (N,h,w) {0,1}
        m = m[keep]
        if retina_masks:
            if m.shape[-2:] != (H, W):
                m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m])
        else:
            m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m])
        masks = m.astype(np.uint8)

    if masks is None or len(mapped) == 0:
        return Detections(
            masks=None,
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=float),
            class_names=target_classes,
        )

    return Detections(
        masks=masks,                 # (N,H,W) {0,1}
        class_id=mapped,             # (N,)
        confidence=conf,             # (N,)
        class_names=target_classes,
    )


__all__ = [
    "Detections",
    "initialize_yoloe_model",
    "set_yoloe_classes",
    "yoloe_detection",
]
