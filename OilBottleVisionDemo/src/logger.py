"""Denetim (inspection) kayıt sistemi ve uygulama log'u."""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import timestamp

INSPECTION_FIELDS: tuple[str, ...] = (
    "timestamp",
    "source_type",
    "source_name",
    "final_status",
    "defect_classes",
    "confidence",
    "inference_time_ms",
    "output_path",
)

_write_lock = threading.Lock()


def get_logger(name: str = "oilbottle", level: int = logging.INFO) -> logging.Logger:
    """Konsola yazan, tekrar eklenmeyen basit bir logger döndürür."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class InspectionLogger:
    """Her kontrolü ``logs/inspections.csv`` dosyasına ekler."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_header(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
                csv.writer(fh).writerow(INSPECTION_FIELDS)

    def log(
        self,
        *,
        source_type: str,
        source_name: str,
        final_status: str,
        defect_classes: Iterable[str] | str = (),
        confidence: float | None = None,
        inference_time_ms: float | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Tek bir kontrol kaydı ekler ve yazılan satırı döndürür."""
        if isinstance(defect_classes, str):
            defects = defect_classes
        else:
            defects = ";".join(str(c) for c in defect_classes)

        row = {
            "timestamp": timestamp(),
            "source_type": source_type,
            "source_name": source_name,
            "final_status": final_status,
            "defect_classes": defects,
            "confidence": "" if confidence is None else f"{float(confidence):.4f}",
            "inference_time_ms": (
                "" if inference_time_ms is None else f"{float(inference_time_ms):.1f}"
            ),
            "output_path": "" if output_path is None else str(output_path),
        }
        with _write_lock:
            self._ensure_header()
            with self.csv_path.open("a", encoding="utf-8-sig", newline="") as fh:
                csv.DictWriter(fh, fieldnames=INSPECTION_FIELDS).writerow(row)
        return row

    def read(self) -> pd.DataFrame:
        """Kayıtları DataFrame olarak döndürür; dosya yoksa boş DataFrame."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return pd.DataFrame(columns=list(INSPECTION_FIELDS))
        try:
            df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        except (pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame(columns=list(INSPECTION_FIELDS))
        for col in INSPECTION_FIELDS:
            if col not in df.columns:
                df[col] = ""
        return df[list(INSPECTION_FIELDS)]

    def clear(self) -> None:
        """Kayıt dosyasını sıfırlar (yalnızca başlık kalır)."""
        with _write_lock:
            with self.csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
                csv.writer(fh).writerow(INSPECTION_FIELDS)
