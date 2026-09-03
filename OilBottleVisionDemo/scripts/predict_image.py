"""Tek görüntü (veya klasör) üzerinde kalite kontrol tahmini.

Kullanım:
    python scripts/predict_image.py --image data/prepared/bottle_qa/test/images/x.jpg
    python scripts/predict_image.py --image klasor/ --json sonuc.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.decision_engine import InspectionResult  # noqa: E402
from src.detector import BottleDetector, ModelLoadError  # noqa: E402
from src.logger import InspectionLogger  # noqa: E402
from src.utils import (  # noqa: E402
    IMAGE_SUFFIXES,
    configure_console,
    file_timestamp,
    sanitize_filename,
    write_json,
)
from src.visualization import annotate_image, read_image, save_annotated  # noqa: E402


def print_result(image_path: Path, result: InspectionResult) -> None:
    """Sonucu terminale Türkçe yazar."""
    bar = "-" * 70
    print(bar)
    print(f"  Dosya   : {image_path.name}")
    print(f"  KARAR   : {result.status}")
    print(f"  Gerekçe : {result.reason}")
    if result.defects:
        print("  Bulunan kusurlar:")
        for det in result.defects:
            print(f"    - {det.label:<28} güven: %{det.confidence * 100:.1f}")
    if result.uncertain:
        print("  Kararsız bant (manuel kontrol önerilir):")
        for det in result.uncertain:
            print(f"    ? {det.label:<28} güven: %{det.confidence * 100:.1f}")
    other = [
        d
        for d in result.detections
        if d not in result.defects and d not in result.uncertain
    ]
    if other:
        print("  Diğer tespitler:")
        for det in other:
            print(f"    · {det.label:<28} güven: %{det.confidence * 100:.1f}")
    if not result.detections:
        print("  Tespit yok.")
    print(f"  Süre    : {result.inference_time_ms:.1f} ms")
    print(bar)


def collect_images(target: Path) -> list[Path]:
    target = Path(target)
    if target.is_file():
        return [target]
    if target.is_dir():
        return [
            p
            for p in sorted(target.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    return []


def main() -> int:
    configure_console()
    config = load_config()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(
        description="OilBottleVisionDemo - görüntü üzerinde kalite kontrol"
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Görüntü dosyası veya klasörü"
    )
    parser.add_argument("--weights", type=Path, default=None, help="Model ağırlığı")
    parser.add_argument("--device", default=None, help="'auto' | 'cpu' | '0'")
    parser.add_argument("--conf", type=float, default=None, help="Tespit eşiği")
    parser.add_argument("--iou", type=float, default=None, help="IoU eşiği")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Annotated görüntülerin kaydedileceği klasör",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Sonuçların yazılacağı JSON dosyası"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Annotated görüntüyü kaydetme"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="logs/inspections.csv kaydını atla"
    )
    args = parser.parse_args()

    images = collect_images(args.image)
    if not images:
        print(f"HATA: Görüntü bulunamadı: {args.image}")
        return 2

    try:
        detector = BottleDetector(
            weights=args.weights,
            device=args.device,
            config=config,
            conf=args.conf,
            iou=args.iou,
        )
    except ModelLoadError as exc:
        print(f"HATA: {exc}")
        return 1

    info = detector.info()
    print()
    print("=" * 70)
    print("  GÖRÜNTÜ ANALİZİ")
    print("=" * 70)
    print(f"  Ağırlık : {info['weights']}"
          + ("" if info["is_trained"] else "  (EĞİTİLMEMİŞ - önce train.py çalıştırın)"))
    print(f"  Cihaz   : {info['device']}")
    print(f"  Eşik    : conf={info['detection_confidence']}  iou={info['iou_threshold']}")
    print(f"  Görüntü : {len(images)} adet")
    print()

    output_dir = Path(args.output_dir) if args.output_dir else config.predictions_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_logger = InspectionLogger(config.inspection_log)

    payload: list[dict[str, Any]] = []
    counters = {"SAĞLAM": 0, "HATALI": 0, "YANLIŞ ÜRÜN": 0, "MANUEL KONTROL GEREKLİ": 0}

    for image_path in images:
        try:
            frame = read_image(image_path)
        except OSError as exc:
            print(f"  ! {image_path.name}: okunamadı ({exc})")
            continue

        result, _ = detector.predict(frame)
        print_result(image_path, result)
        counters[result.status] = counters.get(result.status, 0) + 1

        saved_path: Path | None = None
        if not args.no_save:
            annotated = annotate_image(frame, result)
            # Hatalı ürünler ayrıca outputs/rejected altına kaydedilir.
            target_dir = config.rejected_dir if result.is_defective else output_dir
            stem = sanitize_filename(image_path.stem)
            saved_path = save_annotated(
                annotated, target_dir / f"{stem}_{file_timestamp()}.jpg"
            )
            print(f"  Kaydedildi: {saved_path}")

        if not args.no_log:
            inspection_logger.log(
                source_type="image",
                source_name=image_path.name,
                final_status=result.status,
                defect_classes=result.defect_classes,
                confidence=result.max_confidence,
                inference_time_ms=result.inference_time_ms,
                output_path=saved_path,
            )

        record = result.as_dict()
        record["source"] = str(image_path)
        record["output_path"] = str(saved_path) if saved_path else None
        payload.append(record)

    print()
    print("  ÖZET")
    for status, count in counters.items():
        if count:
            print(f"    {status:<24}: {count}")
    print()

    if args.json:
        write_json(args.json, payload)
        print(f"  JSON çıktı: {args.json}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
