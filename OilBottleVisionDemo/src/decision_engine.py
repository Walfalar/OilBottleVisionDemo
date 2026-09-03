"""Karar motoru.

Model tespitlerini (detections) nihai ürün kararına çevirir.

Kararlar
--------
SAĞLAM                  : Şişe algılandı, kusur yok.
HATALI                  : En az bir kusur sınıfı algılandı.
YANLIŞ ÜRÜN             : Farklı marka şişe algılandı.
MANUEL KONTROL GEREKLİ  : Şişe algılanamadı ya da güven değeri kararsız bantta.

Sınıf İSİMLERİ her zaman veri setinin data.yaml dosyasından gelir; buradaki
eşleme tablosu (configs/project.yaml -> class_mapping) yalnızca bu isimleri
Türkçe açıklamaya ve karar durumuna bağlar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import ClassRule, Config, Thresholds, load_config

# --- Karar anahtarları ve gösterim metinleri ------------------------------
STATUS_OK = "OK"
STATUS_DEFECT = "HATALI"
STATUS_WRONG_PRODUCT = "YANLIS_URUN"
STATUS_MANUAL = "MANUEL"

STATUS_LABELS: dict[str, str] = {
    STATUS_OK: "SAĞLAM",
    STATUS_DEFECT: "HATALI",
    STATUS_WRONG_PRODUCT: "YANLIŞ ÜRÜN",
    STATUS_MANUAL: "MANUEL KONTROL GEREKLİ",
}

# Streamlit / OpenCV görselleştirmesi için ortak renk paleti.
STATUS_COLORS_HEX: dict[str, str] = {
    STATUS_OK: "#1B9E4B",             # yeşil
    STATUS_DEFECT: "#D62828",         # kırmızı
    STATUS_WRONG_PRODUCT: "#F77F00",  # turuncu
    STATUS_MANUAL: "#F2C500",         # sarı
}

# OpenCV BGR karşılıkları
STATUS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    STATUS_OK: (75, 158, 27),
    STATUS_DEFECT: (40, 40, 214),
    STATUS_WRONG_PRODUCT: (0, 127, 247),
    STATUS_MANUAL: (0, 197, 242),
}

# Bilinmeyen (eşleme tablosunda olmayan) sınıflar için varsayılan ciddiyet.
UNKNOWN_SEVERITY = 1


@dataclass
class Detection:
    """Tek bir model tespiti."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "track_id": self.track_id,
        }


@dataclass
class EvaluatedDetection:
    """Karar motorunun sınıflandırdığı tespit."""

    detection: Detection
    rule: ClassRule
    tier: str  # "confident" | "uncertain" | "ignored"

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        return float(self.detection.confidence)

    @property
    def label(self) -> str:
        return self.rule.label

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.detection.bbox

    def as_dict(self) -> dict[str, Any]:
        data = self.detection.as_dict()
        data.update(
            {
                "label": self.rule.label,
                "status": self.rule.status,
                "severity": self.rule.severity,
                "tier": self.tier,
            }
        )
        return data


@dataclass
class InspectionResult:
    """Bir görüntü/frame için nihai kalite kontrol sonucu."""

    status_key: str
    reason: str
    detections: list[EvaluatedDetection] = field(default_factory=list)
    defects: list[EvaluatedDetection] = field(default_factory=list)
    uncertain: list[EvaluatedDetection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    # --- türetilmiş alanlar -------------------------------------------
    @property
    def status(self) -> str:
        """Kullanıcıya gösterilen Türkçe karar metni."""
        return STATUS_LABELS.get(self.status_key, self.status_key)

    @property
    def is_defective(self) -> bool:
        return self.status_key == STATUS_DEFECT

    @property
    def needs_manual_review(self) -> bool:
        return self.status_key == STATUS_MANUAL

    @property
    def color_hex(self) -> str:
        return STATUS_COLORS_HEX.get(self.status_key, "#808080")

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        return STATUS_COLORS_BGR.get(self.status_key, (128, 128, 128))

    @property
    def defect_classes(self) -> list[str]:
        """Bulunan kusur sınıflarının ham adları (tekrarsız, ciddiyete göre)."""
        seen: list[str] = []
        for d in self.defects:
            if d.class_name not in seen:
                seen.append(d.class_name)
        return seen

    @property
    def defect_labels(self) -> list[str]:
        seen: list[str] = []
        for d in self.defects:
            if d.label not in seen:
                seen.append(d.label)
        return seen

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)

    def summary_line(self) -> str:
        """Terminal için tek satırlık Türkçe özet."""
        if self.defect_labels:
            return f"{self.status} - {', '.join(self.defect_labels)}"
        return f"{self.status} - {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_key": self.status_key,
            "reason": self.reason,
            "is_defective": self.is_defective,
            "needs_manual_review": self.needs_manual_review,
            "defect_classes": self.defect_classes,
            "defect_labels": self.defect_labels,
            "max_confidence": round(self.max_confidence, 4),
            "inference_time_ms": round(float(self.inference_time_ms), 1),
            "detections": [d.as_dict() for d in self.detections],
        }


class DecisionEngine:
    """Tespitleri kurallara göre nihai ürün kararına dönüştürür."""

    def __init__(
        self,
        config: Config | None = None,
        thresholds: Thresholds | None = None,
        class_names: Sequence[str] | None = None,
    ):
        self.config = config or load_config()
        self.thresholds = thresholds or self.config.thresholds
        self.rules: dict[str, ClassRule] = self.config.class_rules
        self.class_names: list[str] = list(class_names) if class_names else []

    # --- kural erişimi -------------------------------------------------
    def rule_for(self, class_name: str) -> ClassRule:
        """Sınıf adı için kuralı döndürür.

        Eşleme tablosunda olmayan sınıflar güvenli tarafta kalmak için
        MANUEL KONTROL olarak işaretlenir; asla SAĞLAM sayılmaz.
        """
        rule = self.rules.get(class_name)
        if rule is not None:
            return rule
        return ClassRule(
            name=class_name,
            status=STATUS_MANUAL,
            label=f"Bilinmeyen sınıf ({class_name})",
            severity=UNKNOWN_SEVERITY,
        )

    def unmapped_classes(self, class_names: Iterable[str] | None = None) -> list[str]:
        """data.yaml içinde olup eşleme tablosunda karşılığı olmayan sınıflar."""
        names = list(class_names) if class_names is not None else self.class_names
        return [n for n in names if n not in self.rules]

    # --- tespit sınıflandırma -----------------------------------------
    def _tier(self, confidence: float) -> str:
        t = self.thresholds
        conf = float(confidence)
        if conf < t.manual_review_low:
            return "ignored"
        if conf < t.manual_review_high or conf < t.detection_confidence:
            return "uncertain"
        return "confident"

    # --- ana karar -----------------------------------------------------
    def evaluate(
        self,
        detections: Iterable[Detection],
        inference_time_ms: float = 0.0,
    ) -> InspectionResult:
        """Tespit listesinden nihai kararı üretir."""
        evaluated: list[EvaluatedDetection] = []
        for det in detections:
            rule = self.rule_for(det.class_name)
            evaluated.append(
                EvaluatedDetection(
                    detection=det, rule=rule, tier=self._tier(det.confidence)
                )
            )

        kept = [e for e in evaluated if e.tier != "ignored"]
        confident = [e for e in kept if e.tier == "confident"]
        uncertain = [e for e in kept if e.tier == "uncertain"]

        # Ciddiyeti yüksek olan başta olacak şekilde sıralama anahtarı.
        def _sort_key(e: EvaluatedDetection) -> tuple[int, float]:
            return (e.rule.severity, e.confidence)

        confident_defects = sorted(
            (e for e in confident if e.rule.status == STATUS_DEFECT),
            key=_sort_key,
            reverse=True,
        )
        uncertain_defects = sorted(
            (e for e in uncertain if e.rule.status == STATUS_DEFECT),
            key=_sort_key,
            reverse=True,
        )

        # 1) Hiçbir anlamlı tespit yok -> manuel kontrol
        if not kept:
            return InspectionResult(
                status_key=STATUS_MANUAL,
                reason="Şişe algılanamadı",
                detections=[],
                defects=[],
                uncertain=[],
                inference_time_ms=inference_time_ms,
            )

        # 2) Güvenilir kusur -> HATALI (en ciddi kusur karara adını verir)
        if confident_defects:
            top = confident_defects[0]
            return InspectionResult(
                status_key=STATUS_DEFECT,
                reason=top.label,
                detections=kept,
                defects=confident_defects,
                uncertain=uncertain,
                inference_time_ms=inference_time_ms,
            )

        # 3) Güvenilir tespitler arasında en ciddi olanı belirle
        if confident:
            top = max(confident, key=_sort_key)
            if top.rule.status == STATUS_WRONG_PRODUCT:
                return InspectionResult(
                    status_key=STATUS_WRONG_PRODUCT,
                    reason=top.label,
                    detections=kept,
                    defects=[],
                    uncertain=uncertain,
                    inference_time_ms=inference_time_ms,
                )
            if top.rule.status == STATUS_MANUAL:
                return InspectionResult(
                    status_key=STATUS_MANUAL,
                    reason=top.label,
                    detections=kept,
                    defects=[],
                    uncertain=uncertain,
                    inference_time_ms=inference_time_ms,
                )

        # 4) Kararsız bantta tespit varsa güvenli taraf: manuel kontrol
        if uncertain:
            focus = uncertain_defects[0] if uncertain_defects else max(uncertain, key=_sort_key)
            reason = (
                f"Kararsız güven değeri: {focus.label} "
                f"(%{focus.confidence * 100:.0f})"
            )
            return InspectionResult(
                status_key=STATUS_MANUAL,
                reason=reason,
                detections=kept,
                defects=[],
                uncertain=uncertain,
                inference_time_ms=inference_time_ms,
            )

        # 5) Şişe var, kusur yok -> SAĞLAM
        return InspectionResult(
            status_key=STATUS_OK,
            reason="Kusur tespit edilmedi",
            detections=kept,
            defects=[],
            uncertain=[],
            inference_time_ms=inference_time_ms,
        )

    # --- yardımcı ------------------------------------------------------
    def evaluate_raw(
        self,
        raw: Iterable[tuple[int, str, float, tuple[float, float, float, float]]],
        inference_time_ms: float = 0.0,
    ) -> InspectionResult:
        """(class_id, class_name, confidence, bbox) demetlerinden karar üretir."""
        dets = [
            Detection(class_id=c_id, class_name=name, confidence=conf, bbox=bbox)
            for c_id, name, conf, bbox in raw
        ]
        return self.evaluate(dets, inference_time_ms=inference_time_ms)
