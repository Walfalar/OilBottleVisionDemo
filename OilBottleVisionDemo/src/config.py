"""Merkezi yapılandırma yükleyici.

Tüm yollar proje köküne göre pathlib ile çözülür; Windows ve Linux'ta
aynı şekilde çalışır.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src -> proje kökü
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "project.yaml"


def _deep_get(data: dict, dotted: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass
class Thresholds:
    """Karar motorunun kullandığı eşikler."""

    detection_confidence: float = 0.25
    manual_review_low: float = 0.15
    manual_review_high: float = 0.30
    iou_threshold: float = 0.45

    def validate(self) -> list[str]:
        """Tutarsız eşikleri uyarı listesi olarak döndürür."""
        warnings: list[str] = []
        for name in (
            "detection_confidence",
            "manual_review_low",
            "manual_review_high",
            "iou_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                warnings.append(f"{name} 0-1 aralığında olmalı (şu an: {value}).")
        if self.manual_review_low > self.manual_review_high:
            warnings.append(
                "manual_review_low, manual_review_high değerinden büyük olamaz."
            )
        return warnings


@dataclass
class ClassRule:
    """Bir model sınıfı için karar kuralı."""

    name: str
    status: str
    label: str
    severity: int

    @property
    def is_defect(self) -> bool:
        return self.status == "HATALI"


@dataclass
class Config:
    """Proje yapılandırmasının tip güvenli görünümü."""

    raw: dict = field(default_factory=dict)
    path: Path = DEFAULT_CONFIG_PATH
    root: Path = PROJECT_ROOT

    # --- yollar -------------------------------------------------------
    def resolve(self, key: str, fallback: str) -> Path:
        rel = _deep_get(self.raw, f"paths.{key}", fallback) or fallback
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data_source_dir(self) -> Path:
        return self.resolve("data_source", "data/source")

    @property
    def data_prepared_dir(self) -> Path:
        return self.resolve("data_prepared", "data/prepared")

    @property
    def models_dir(self) -> Path:
        return self.resolve("models", "models")

    @property
    def training_output_dir(self) -> Path:
        return self.resolve("training_output", "outputs/training")

    @property
    def predictions_dir(self) -> Path:
        return self.resolve("predictions_output", "outputs/predictions")

    @property
    def rejected_dir(self) -> Path:
        return self.resolve("rejected_output", "outputs/rejected")

    @property
    def reports_dir(self) -> Path:
        return self.resolve("reports_output", "outputs/reports")

    @property
    def logs_dir(self) -> Path:
        return self.resolve("logs", "logs")

    @property
    def inspection_log(self) -> Path:
        return self.resolve("inspection_log", "logs/inspections.csv")

    # --- model --------------------------------------------------------
    @property
    def model_weights(self) -> str:
        return str(_deep_get(self.raw, "model.weights", "yolo26n.pt"))

    @property
    def fallback_weights(self) -> str:
        return str(_deep_get(self.raw, "model.fallback_weights", "yolo11n.pt"))

    @property
    def trained_weights(self) -> Path:
        rel = _deep_get(self.raw, "model.trained_weights", "models/best.pt")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def last_weights(self) -> Path:
        rel = _deep_get(self.raw, "model.last_weights", "models/last.pt")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def imgsz(self) -> int:
        return int(_deep_get(self.raw, "model.imgsz", 640))

    @property
    def device_setting(self) -> str:
        return str(_deep_get(self.raw, "model.device", "auto"))

    # --- eşikler ------------------------------------------------------
    @property
    def thresholds(self) -> Thresholds:
        node = _deep_get(self.raw, "thresholds", {}) or {}
        return Thresholds(
            detection_confidence=float(node.get("detection_confidence", 0.25)),
            manual_review_low=float(node.get("manual_review_low", 0.15)),
            manual_review_high=float(node.get("manual_review_high", 0.30)),
            iou_threshold=float(node.get("iou_threshold", 0.45)),
        )

    # --- eğitim -------------------------------------------------------
    @property
    def training(self) -> dict:
        return copy.deepcopy(_deep_get(self.raw, "training", {}) or {})

    @property
    def augment(self) -> dict:
        return copy.deepcopy(_deep_get(self.raw, "training.augment", {}) or {})

    @property
    def video(self) -> dict:
        return copy.deepcopy(_deep_get(self.raw, "video", {}) or {})

    # --- sınıf kuralları ----------------------------------------------
    @property
    def class_rules(self) -> dict[str, ClassRule]:
        node = _deep_get(self.raw, "class_mapping", {}) or {}
        rules: dict[str, ClassRule] = {}
        for name, cfg in node.items():
            cfg = cfg or {}
            rules[name] = ClassRule(
                name=name,
                status=str(cfg.get("status", "MANUEL")),
                label=str(cfg.get("label", name)),
                severity=int(cfg.get("severity", 0)),
            )
        return rules

    def ensure_dirs(self) -> None:
        """Çıktı klasörlerini oluşturur (idempotent)."""
        for d in (
            self.data_source_dir,
            self.data_prepared_dir,
            self.models_dir,
            self.training_output_dir,
            self.predictions_dir,
            self.rejected_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Yapılandırmayı yükler.

    Sıra: açık parametre > OILBOTTLE_CONFIG ortam değişkeni > varsayılan.
    Dosya yoksa boş sözlükle çalışır ve tüm varsayılanlar devreye girer.
    """
    if path is None:
        env = os.environ.get("OILBOTTLE_CONFIG")
        path = Path(env) if env else DEFAULT_CONFIG_PATH
    path = Path(path)
    raw: dict = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    root = path.resolve().parent.parent if path.is_file() else PROJECT_ROOT
    return Config(raw=raw, path=path, root=root)
