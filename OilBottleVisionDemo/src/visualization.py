"""Görselleştirme: bounding box çizimi ve karar rozeti.

Türkçe karakterlerin doğru görünmesi için metinler OpenCV yerine Pillow ile
çizilir (cv2.putText yalnızca ASCII destekler).
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .decision_engine import (
    STATUS_COLORS_BGR,
    STATUS_DEFECT,
    STATUS_MANUAL,
    STATUS_OK,
    STATUS_WRONG_PRODUCT,
    EvaluatedDetection,
    InspectionResult,
)

# Kararsız (manuel bant) tespitler için kesikli/soluk görünüm
UNCERTAIN_BGR = (0, 197, 242)

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


@functools.lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Türkçe karakterleri destekleyen bir TrueType font döndürür."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # matplotlib her zaman DejaVuSans ile birlikte gelir.
    try:
        import matplotlib  # noqa: PLC0415

        dejavu = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if dejavu.is_file():
            return ImageFont.truetype(str(dejavu), size)
    except ImportError:
        pass
    return ImageFont.load_default()


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    return (r, g, b)


def detection_color_bgr(det: EvaluatedDetection) -> tuple[int, int, int]:
    """Tek bir tespitin kutu rengini kuralına göre belirler."""
    if det.tier == "uncertain":
        return UNCERTAIN_BGR
    return STATUS_COLORS_BGR.get(det.rule.status, (128, 128, 128))


def _draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    bg_rgb: tuple[int, int, int],
    fg_rgb: tuple[int, int, int] = (255, 255, 255),
    padding: int = 4,
) -> tuple[float, float, float, float]:
    """Arka planı dolu bir metin kutusu çizer; kutunun sınırlarını döndürür."""
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rect = (x, y, x + tw + 2 * padding, y + th + 2 * padding)
    draw.rectangle(rect, fill=bg_rgb)
    draw.text((x + padding - bbox[0], y + padding - bbox[1]), text, font=font, fill=fg_rgb)
    return rect


def annotate_image(
    image_bgr: np.ndarray,
    result: InspectionResult,
    show_banner: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Tespit kutularını ve karar rozetini çizilmiş BGR görüntü döndürür."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Boş görüntü çizilemez.")

    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))

    # 1) Kutuları OpenCV ile çiz (hızlı)
    for det in result.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        color = detection_color_bgr(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    # 2) Metinleri Pillow ile çiz (Türkçe karakter desteği)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    label_size = max(13, int(round(min(h, w) / 45)))
    font = _load_font(label_size)

    for det in result.detections:
        x1, y1, _, _ = (int(round(v)) for v in det.bbox)
        text = det.label
        if show_confidence:
            text = f"{det.label}  %{det.confidence * 100:.0f}"
        if det.tier == "uncertain":
            text = f"? {text}"
        bg = _bgr_to_rgb(detection_color_bgr(det))
        fg = (0, 0, 0) if det.tier == "uncertain" else (255, 255, 255)
        ty = max(0, y1 - label_size - 10)
        _draw_text_with_bg(draw, (max(0, x1), ty), text, font, bg, fg)

    if show_banner:
        banner_size = max(18, int(round(min(h, w) / 26)))
        banner_font = _load_font(banner_size)
        banner_text = result.status
        if result.defect_labels:
            banner_text = f"{result.status}: {', '.join(result.defect_labels)}"
        elif result.status_key == STATUS_MANUAL:
            banner_text = f"{result.status} ({result.reason})"
        bg = _bgr_to_rgb(result.color_bgr)
        _draw_text_with_bg(draw, (8, 8), banner_text, banner_font, bg, (255, 255, 255), padding=8)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save_annotated(
    image_bgr: np.ndarray,
    output_path: Path,
) -> Path:
    """Görüntüyü diske yazar (Türkçe/Unicode yol destekli)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".jpg"
    ok, buffer = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise OSError(f"Görüntü kodlanamadı: {output_path}")
    output_path.write_bytes(buffer.tobytes())
    return output_path


def read_image(path: Path) -> np.ndarray:
    """Unicode yol destekli görüntü okuma (cv2.imread Windows'ta sorunlu)."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Görüntü okunamadı: {path}")
    return image


def status_legend() -> dict[str, str]:
    """Arayüzde gösterilecek renk açıklamaları."""
    return {
        STATUS_OK: "Yeşil - Sağlam",
        STATUS_DEFECT: "Kırmızı - Hatalı",
        STATUS_WRONG_PRODUCT: "Turuncu - Yanlış ürün",
        STATUS_MANUAL: "Sarı - Manuel kontrol",
    }
