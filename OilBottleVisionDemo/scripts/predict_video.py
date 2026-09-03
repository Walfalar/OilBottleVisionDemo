"""Video üzerinde kalite kontrol.

Aynı şişe ardışık framelerde tekrar tekrar sayılmasın diye IoU tabanlı basit
bir takip + zaman tabanlı tekrar engelleme kullanılır.

Kullanım:
    python scripts/predict_video.py --video hat.mp4
    python scripts/predict_video.py --video hat.mp4 --stride 2 --no-video-out
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.decision_engine import (  # noqa: E402
    STATUS_DEFECT,
    STATUS_MANUAL,
    STATUS_OK,
    STATUS_WRONG_PRODUCT,
    InspectionResult,
)
from src.detector import BottleDetector, ModelLoadError  # noqa: E402
from src.logger import InspectionLogger  # noqa: E402
from src.utils import configure_console, file_timestamp, sanitize_filename, write_json  # noqa: E402
from src.visualization import annotate_image, save_annotated  # noqa: E402

# Kararların "ciddiyet" sırası: bir ürün için en kötü karar geçerli olur.
STATUS_PRIORITY: dict[str, int] = {
    STATUS_DEFECT: 3,
    STATUS_WRONG_PRODUCT: 2,
    STATUS_MANUAL: 1,
    STATUS_OK: 0,
}


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Kutunun merkez noktası."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """İki kutunun merkezleri arasındaki öklid mesafesi."""
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """İki kutu arasındaki IoU değeri."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ProductRecord:
    """Video boyunca sayılan tek bir ürün."""

    product_id: int
    first_time_s: float
    last_time_s: float
    status_key: str
    reason: str
    defect_classes: list[str] = field(default_factory=list)
    defect_labels: list[str] = field(default_factory=list)
    max_confidence: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    frame_count: int = 0
    best_frame_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        from src.decision_engine import STATUS_LABELS  # noqa: PLC0415

        return {
            "product_id": self.product_id,
            "first_time_s": round(self.first_time_s, 2),
            "last_time_s": round(self.last_time_s, 2),
            "status": STATUS_LABELS.get(self.status_key, self.status_key),
            "status_key": self.status_key,
            "reason": self.reason,
            "defect_classes": ";".join(self.defect_classes),
            "defect_labels": ";".join(self.defect_labels),
            "max_confidence": round(self.max_confidence, 4),
            "frame_count": self.frame_count,
            "best_frame_index": self.best_frame_index,
        }


class ProductTracker:
    """Ardışık framelerdeki aynı şişeyi tek ürün olarak sayar.

    Üç kriter birlikte kullanılır:
      - Zaman: son görülmeden bu yana geçen süre eşiği aşarsa yeni üründür.
      - IoU: kutular örtüşüyorsa aynı ürün.
      - Merkez mesafesi: hızlı hatlarda kutu kendi boyutundan fazla yer
        değiştirip IoU'yu sıfırlayabilir; merkezler birbirine yakınsa tespit
        yine aynı ürüne bağlanır.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        suppression_sec: float = 1.5,
        center_distance_ratio: float = 0.15,
        frame_size: tuple[int, int] = (640, 640),
    ):
        self.iou_threshold = float(iou_threshold)
        self.suppression_sec = float(suppression_sec)
        width, height = frame_size
        diagonal = float((max(width, 1) ** 2 + max(height, 1) ** 2) ** 0.5)
        self.max_center_distance = float(center_distance_ratio) * diagonal
        self.products: list[ProductRecord] = []
        self._active: ProductRecord | None = None
        self._next_id = 1

    @staticmethod
    def _primary_bbox(result: InspectionResult) -> tuple[float, float, float, float]:
        """Takibin dayanacağı çıpa kutusunu döndürür.

        Çıpa olarak TÜM tespitlerin en büyüğü (tipik olarak şişenin kendisi)
        seçilir. Kusur kutusu küçük olduğu için ardışık framelerde örtüşmez ve
        aynı ürünün birden çok kez sayılmasına yol açardı.
        """
        if not result.detections:
            return (0.0, 0.0, 0.0, 0.0)

        def area(det) -> float:
            x1, y1, x2, y2 = det.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return max(result.detections, key=area).bbox

    def update(self, result: InspectionResult, time_s: float, frame_index: int) -> None:
        """Bir frame sonucunu takipçiye işler."""
        if not result.detections:
            # Şişe yok: aktif ürün hattan çıkmış sayılır.
            if self._active is not None and (
                time_s - self._active.last_time_s > self.suppression_sec
            ):
                self._active = None
            return

        bbox = self._primary_bbox(result)
        same_product = False
        if self._active is not None:
            gap = time_s - self._active.last_time_s
            if gap <= self.suppression_sec:
                overlap = iou(self._active.bbox, bbox)
                distance = center_distance(self._active.bbox, bbox)
                same_product = (
                    overlap >= self.iou_threshold
                    or distance <= self.max_center_distance
                )

        if same_product and self._active is not None:
            record = self._active
            record.last_time_s = time_s
            record.frame_count += 1
            record.bbox = bbox
            # Ürün için EN KÖTÜ karar geçerli olur (kusuru kaçırmamak için).
            if STATUS_PRIORITY.get(result.status_key, 0) > STATUS_PRIORITY.get(
                record.status_key, 0
            ):
                record.status_key = result.status_key
                record.reason = result.reason
                record.best_frame_index = frame_index
            for cls, label in zip(result.defect_classes, result.defect_labels):
                if cls not in record.defect_classes:
                    record.defect_classes.append(cls)
                    record.defect_labels.append(label)
            record.max_confidence = max(record.max_confidence, result.max_confidence)
            return

        record = ProductRecord(
            product_id=self._next_id,
            first_time_s=time_s,
            last_time_s=time_s,
            status_key=result.status_key,
            reason=result.reason,
            defect_classes=list(result.defect_classes),
            defect_labels=list(result.defect_labels),
            max_confidence=result.max_confidence,
            bbox=bbox,
            frame_count=1,
            best_frame_index=frame_index,
        )
        self._next_id += 1
        self.products.append(record)
        self._active = record

    def summary(self) -> dict[str, Any]:
        from src.decision_engine import STATUS_LABELS  # noqa: PLC0415

        status_counts: Counter = Counter(p.status_key for p in self.products)
        defect_counts: Counter = Counter()
        for p in self.products:
            for label in p.defect_labels:
                defect_counts[label] += 1
        return {
            "total_products": len(self.products),
            "ok": status_counts.get(STATUS_OK, 0),
            "defective": status_counts.get(STATUS_DEFECT, 0),
            "wrong_product": status_counts.get(STATUS_WRONG_PRODUCT, 0),
            "manual_review": status_counts.get(STATUS_MANUAL, 0),
            "status_counts": {
                STATUS_LABELS.get(k, k): v for k, v in status_counts.items()
            },
            "defect_distribution": dict(defect_counts),
        }


def process_video(
    video_path: Path,
    detector: BottleDetector,
    config: Config,
    output_video: Path | None = None,
    stride: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    save_rejected_frames: bool = True,
) -> dict[str, Any]:
    """Videoyu frame bazlı analiz eder ve özet döndürür.

    Streamlit arayüzü de bu fonksiyonu kullanır.
    """
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Video açılamadı: {video_path}")

    video_cfg = config.video
    stride = int(stride or video_cfg.get("frame_stride", 3))
    stride = max(1, stride)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps != fps:  # NaN koruması
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    if output_video is not None:
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = None

    tracker = ProductTracker(
        iou_threshold=float(video_cfg.get("iou_track_threshold", 0.35)),
        suppression_sec=float(video_cfg.get("duplicate_suppression_sec", 1.5)),
        center_distance_ratio=float(video_cfg.get("max_center_distance_ratio", 0.15)),
        frame_size=(width or 640, height or 640),
    )

    frame_rows: list[dict[str, Any]] = []
    rejected_saved: list[str] = []
    saved_product_ids: set[int] = set()
    frame_index = 0
    analyzed = 0
    total_inference_ms = 0.0
    last_result: InspectionResult | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            time_s = frame_index / fps
            if frame_index % stride == 0:
                result, _ = detector.predict(frame)
                last_result = result
                analyzed += 1
                total_inference_ms += result.inference_time_ms
                tracker.update(result, time_s, frame_index)
                frame_rows.append(
                    {
                        "frame": frame_index,
                        "time_s": round(time_s, 2),
                        "status": result.status,
                        "defect_classes": ";".join(result.defect_classes),
                        "max_confidence": round(result.max_confidence, 4),
                        "inference_time_ms": round(result.inference_time_ms, 1),
                    }
                )

                # Her hatalı ürün için tek bir kare kaydedilir.
                if (
                    save_rejected_frames
                    and result.is_defective
                    and tracker.products
                    and tracker.products[-1].product_id not in saved_product_ids
                ):
                    product = tracker.products[-1]
                    saved_product_ids.add(product.product_id)
                    stem = sanitize_filename(video_path.stem)
                    path = save_annotated(
                        annotate_image(frame, result),
                        config.rejected_dir
                        / f"{stem}_urun{product.product_id:04d}_frame{frame_index}.jpg",
                    )
                    rejected_saved.append(str(path))

            if writer is not None:
                drawn = (
                    annotate_image(frame, last_result)
                    if last_result is not None
                    else frame
                )
                writer.write(drawn)

            frame_index += 1
            if progress_callback and total_frames:
                progress_callback(frame_index, total_frames)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    summary = tracker.summary()
    summary.update(
        {
            "source": str(video_path),
            "fps": round(float(fps), 2),
            "resolution": f"{width}x{height}",
            "total_frames": frame_index,
            "analyzed_frames": analyzed,
            "frame_stride": stride,
            "avg_inference_ms": round(total_inference_ms / analyzed, 1) if analyzed else 0.0,
            "total_inference_ms": round(total_inference_ms, 1),
            "output_video": str(output_video) if writer is not None else None,
            "rejected_frames": rejected_saved,
        }
    )
    return {
        "summary": summary,
        "products": [p.as_dict() for p in tracker.products],
        "frames": frame_rows,
    }


def main() -> int:
    configure_console()
    config = load_config()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(
        description="OilBottleVisionDemo - video üzerinde kalite kontrol"
    )
    parser.add_argument("--video", type=Path, required=True, help="Video dosyası (mp4)")
    parser.add_argument("--weights", type=Path, default=None, help="Model ağırlığı")
    parser.add_argument("--device", default=None, help="'auto' | 'cpu' | '0'")
    parser.add_argument("--conf", type=float, default=None, help="Tespit eşiği")
    parser.add_argument("--iou", type=float, default=None, help="IoU eşiği")
    parser.add_argument("--stride", type=int, default=None, help="Her N. frame analiz")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="En fazla işlenecek frame"
    )
    parser.add_argument("--output", type=Path, default=None, help="Çıktı MP4 yolu")
    parser.add_argument(
        "--no-video-out", action="store_true", help="Annotated video üretme"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="logs/inspections.csv kaydını atla"
    )
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"HATA: Video bulunamadı: {args.video}")
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
    stamp = file_timestamp()
    stem = sanitize_filename(args.video.stem)
    output_video = None
    if not args.no_video_out:
        output_video = args.output or (
            config.predictions_dir / f"{stem}_{stamp}_analiz.mp4"
        )

    print()
    print("=" * 70)
    print("  VİDEO ANALİZİ")
    print("=" * 70)
    print(f"  Video   : {args.video}")
    print(f"  Ağırlık : {info['weights']}")
    print(f"  Cihaz   : {info['device']}")
    print()

    last_percent = -1

    def progress(current: int, total: int) -> None:
        nonlocal last_percent
        percent = int(100 * current / max(total, 1))
        if percent != last_percent and percent % 5 == 0:
            last_percent = percent
            print(f"  İşleniyor... %{percent}", end="\r", flush=True)

    try:
        result = process_video(
            args.video,
            detector,
            config,
            output_video=output_video,
            stride=args.stride,
            progress_callback=progress,
            max_frames=args.max_frames,
        )
    except OSError as exc:
        print(f"HATA: {exc}")
        return 1

    summary = result["summary"]
    print(" " * 40, end="\r")
    print("  ÖZET")
    print(f"    Çözünürlük / FPS        : {summary['resolution']} @ {summary['fps']}")
    print(f"    Toplam frame            : {summary['total_frames']}")
    print(f"    Analiz edilen frame     : {summary['analyzed_frames']} "
          f"(her {summary['frame_stride']}. frame)")
    print(f"    Ortalama inference      : {summary['avg_inference_ms']} ms")
    print()
    print(f"    TOPLAM İNCELENEN ÜRÜN   : {summary['total_products']}")
    print(f"    Sağlam                  : {summary['ok']}")
    print(f"    Hatalı                  : {summary['defective']}")
    print(f"    Yanlış ürün             : {summary['wrong_product']}")
    print(f"    Manuel kontrol          : {summary['manual_review']}")
    if summary["defect_distribution"]:
        print("    Kusur dağılımı:")
        for label, count in sorted(
            summary["defect_distribution"].items(), key=lambda kv: -kv[1]
        ):
            print(f"      - {label:<28}: {count}")
    print()

    # --- çıktılar --------------------------------------------------------
    base = config.predictions_dir / f"{stem}_{stamp}"
    json_path = write_json(base.with_name(base.name + "_ozet.json"), result)
    csv_path = base.with_name(base.name + "_urunler.csv")
    pd.DataFrame(result["products"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    frames_csv = base.with_name(base.name + "_frameler.csv")
    pd.DataFrame(result["frames"]).to_csv(frames_csv, index=False, encoding="utf-8-sig")

    if summary["output_video"]:
        print(f"  Analiz videosu : {summary['output_video']}")
    print(f"  Özet JSON      : {json_path}")
    print(f"  Ürün CSV       : {csv_path}")
    print(f"  Frame CSV      : {frames_csv}")
    if summary["rejected_frames"]:
        print(f"  Hatalı kareler : {config.rejected_dir} "
              f"({len(summary['rejected_frames'])} adet)")
    print()

    if not args.no_log:
        logger = InspectionLogger(config.inspection_log)
        for product in result["products"]:
            logger.log(
                source_type="video",
                source_name=f"{args.video.name}#urun{product['product_id']}",
                final_status=product["status"],
                defect_classes=product["defect_classes"],
                confidence=product["max_confidence"],
                inference_time_ms=summary["avg_inference_ms"],
                output_path=summary["output_video"] or "",
            )
        print(f"  Kayıtlar: {config.inspection_log}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
