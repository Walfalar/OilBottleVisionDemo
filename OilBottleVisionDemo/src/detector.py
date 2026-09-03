"""YOLO tabanlı şişe tespit sarmalayıcısı.

Model seçimi modülerdir: ``configs/project.yaml`` içindeki ``model.weights``
değeri yüklenemezse (ör. kurulu Ultralytics sürümü YOLO26'yı desteklemiyorsa)
otomatik olarak ``model.fallback_weights`` denenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config, load_config
from .decision_engine import DecisionEngine, Detection, InspectionResult
from .logger import get_logger
from .utils import resolve_device, stopwatch, torch_device_string

logger = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Hiçbir model ağırlığı yüklenemediğinde fırlatılır."""


@dataclass
class LoadedModel:
    """Yüklenen modelin bilgileri."""

    model: Any
    weights: str
    device: str
    class_names: list[str]
    was_fallback: bool = False
    is_trained: bool = False


def _extract_class_names(model: Any) -> list[str]:
    """Ultralytics model nesnesinden sınıf isimlerini index sırasıyla alır."""
    names = getattr(model, "names", None)
    if names is None:
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def load_model(
    weights: str | Path | None = None,
    device: str | None = None,
    config: Config | None = None,
) -> LoadedModel:
    """Modeli yükler; başarısız olursa yedek ağırlığa düşer.

    Ağırlık seçim sırası (weights verilmediyse):
      1. models/best.pt (eğitilmiş model varsa)
      2. configs/project.yaml -> model.weights (varsayılan yolo26n.pt)
      3. configs/project.yaml -> model.fallback_weights (yolo11n.pt)
    """
    from ultralytics import YOLO  # noqa: PLC0415 - ağır import, gecikmeli yüklenir

    config = config or load_config()
    resolved_device = resolve_device(device or config.device_setting)

    candidates: list[tuple[str, bool]] = []  # (ağırlık, eğitilmiş mi)
    if weights is not None:
        candidates.append((str(weights), Path(weights).exists()))
    else:
        if config.trained_weights.is_file():
            candidates.append((str(config.trained_weights), True))
        candidates.append((config.model_weights, False))
    fallback = config.fallback_weights
    if all(c[0] != fallback for c in candidates):
        candidates.append((fallback, False))

    errors: list[str] = []
    for idx, (weight, is_trained) in enumerate(candidates):
        try:
            model = YOLO(weight)
            names = _extract_class_names(model)
            if resolved_device != "cpu":
                try:
                    model.to(torch_device_string(resolved_device))
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        "GPU'ya taşınamadı (%s); CPU kullanılacak.", exc
                    )
                    resolved_device = "cpu"
            return LoadedModel(
                model=model,
                weights=weight,
                device=resolved_device,
                class_names=names,
                was_fallback=idx > 0,
                is_trained=is_trained,
            )
        except Exception as exc:  # noqa: BLE001 - ultralytics çeşitli hatalar atar
            errors.append(f"{weight}: {exc}")
            logger.warning("Model yüklenemedi (%s): %s", weight, exc)

    raise ModelLoadError(
        "Hiçbir model ağırlığı yüklenemedi.\n  " + "\n  ".join(errors)
    )


class BottleDetector:
    """Şişe tespiti + karar motoru sarmalayıcısı."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        config: Config | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ):
        self.config = config or load_config()
        self.loaded = load_model(weights=weights, device=device, config=self.config)
        thresholds = self.config.thresholds
        if conf is not None:
            thresholds.detection_confidence = float(conf)
        if iou is not None:
            thresholds.iou_threshold = float(iou)
        self.thresholds = thresholds
        self.engine = DecisionEngine(
            config=self.config,
            thresholds=thresholds,
            class_names=self.loaded.class_names,
        )

    # --- bilgi ---------------------------------------------------------
    @property
    def class_names(self) -> list[str]:
        return self.loaded.class_names

    @property
    def device(self) -> str:
        return self.loaded.device

    @property
    def weights(self) -> str:
        return self.loaded.weights

    def info(self) -> dict[str, Any]:
        return {
            "weights": self.loaded.weights,
            "device": self.loaded.device,
            "is_trained": self.loaded.is_trained,
            "was_fallback": self.loaded.was_fallback,
            "class_names": self.class_names,
            "detection_confidence": self.thresholds.detection_confidence,
            "iou_threshold": self.thresholds.iou_threshold,
            "manual_review_low": self.thresholds.manual_review_low,
            "manual_review_high": self.thresholds.manual_review_high,
            "unmapped_classes": self.engine.unmapped_classes(self.class_names),
        }

    # --- tespit --------------------------------------------------------
    def _to_detections(self, result: Any) -> list[Detection]:
        """Ultralytics sonucunu Detection listesine çevirir."""
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        names = _extract_class_names(result) or self.class_names
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        track_ids = None
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(classes)):
            cid = int(classes[i])
            name = names[cid] if 0 <= cid < len(names) else f"id_{cid}"
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(confs[i]),
                    bbox=tuple(float(v) for v in xyxy[i]),
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
            )
        return detections

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> tuple[InspectionResult, list[Detection]]:
        """Tek bir görüntü/frame için karar ve ham tespitleri döndürür."""
        # Karar motoru kararsız bandı da değerlendirebilsin diye modele
        # manual_review_low eşiği verilir; filtreleme karar motorunda yapılır.
        model_conf = min(
            float(conf if conf is not None else self.thresholds.detection_confidence),
            float(self.thresholds.manual_review_low),
        )
        model_iou = float(iou if iou is not None else self.thresholds.iou_threshold)

        with stopwatch() as t:
            results = self.loaded.model.predict(
                source=source,
                conf=model_conf,
                iou=model_iou,
                imgsz=self.config.imgsz,
                device=self.loaded.device,
                verbose=False,
            )
        detections = self._to_detections(results[0]) if results else []
        decision = self.engine.evaluate(detections, inference_time_ms=t["ms"])
        return decision, detections

    def predict_batch(
        self,
        sources: Sequence[str | Path | np.ndarray],
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[tuple[InspectionResult, list[Detection]]]:
        """Birden fazla görüntüyü tek çağrıda işler."""
        model_conf = min(
            float(conf if conf is not None else self.thresholds.detection_confidence),
            float(self.thresholds.manual_review_low),
        )
        model_iou = float(iou if iou is not None else self.thresholds.iou_threshold)

        with stopwatch() as t:
            results = self.loaded.model.predict(
                source=list(sources),
                conf=model_conf,
                iou=model_iou,
                imgsz=self.config.imgsz,
                device=self.loaded.device,
                verbose=False,
            )
        per_item_ms = t["ms"] / max(len(results), 1)
        output: list[tuple[InspectionResult, list[Detection]]] = []
        for res in results:
            dets = self._to_detections(res)
            output.append((self.engine.evaluate(dets, per_item_ms), dets))
        return output
