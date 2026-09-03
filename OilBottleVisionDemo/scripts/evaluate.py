"""Model değerlendirme.

Kalite kontrolde en kritik hata, KUSURLU ürünün sağlam kabul edilmesidir.
Bu nedenle genel accuracy yerine sınıf bazlı RECALL öne çıkarılır.

Üretilen çıktılar (outputs/reports altında):
  - evaluation_report.json      : tüm metrikler
  - evaluation_per_class.csv    : sınıf bazlı precision/recall/F1/mAP
  - confusion_matrix*.png       : Ultralytics karışıklık matrisi
  - false_positives.csv         : yanlış pozitif örnekler
  - missed_defects.csv          : KAÇIRILMIŞ kusur örnekleri (en kritik)

Kullanım:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights models/best.pt --split test
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    dataset_splits_present,
    labels_dir_for,
    read_class_names,
    read_data_yaml,
    resolve_split_dir,
)
from src.decision_engine import STATUS_DEFECT, DecisionEngine  # noqa: E402
from src.detector import BottleDetector  # noqa: E402
from src.utils import (  # noqa: E402
    IMAGE_SUFFIXES,
    configure_console,
    resolve_device,
    write_json,
)
from scripts.train import extract_metrics, locate_data_yaml  # noqa: E402


# ==========================================================================
# Ultralytics val() metrikleri
# ==========================================================================
def per_class_table(metrics: Any, class_names: list[str]) -> pd.DataFrame:
    """Sınıf bazlı precision / recall / F1 / mAP tablosu üretir."""
    box = getattr(metrics, "box", None)
    if box is None:
        return pd.DataFrame()

    # class_result(i) -> (precision, recall, ap50, ap)
    raw_indices = getattr(box, "ap_class_index", None)
    # numpy dizisi gelebilir; "or []" kullanmak ValueError verir.
    indices = [] if raw_indices is None else list(raw_indices)
    rows: list[dict[str, Any]] = []
    for i, class_idx in enumerate(indices):
        try:
            p, r, ap50, ap = box.class_result(i)
        except (IndexError, ValueError):
            continue
        name = (
            class_names[int(class_idx)]
            if 0 <= int(class_idx) < len(class_names)
            else f"id_{class_idx}"
        )
        p, r = float(p), float(r)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        rows.append(
            {
                "class_name": name,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(run_dir: Path, reports_dir: Path) -> list[Path]:
    """Ultralytics'in ürettiği confusion matrix görsellerini rapor klasörüne kopyalar."""
    copied: list[Path] = []
    for pattern in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        source = run_dir / pattern
        if source.is_file():
            target = reports_dir / pattern
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def confusion_matrix_to_csv(
    metrics: Any, class_names: list[str], output_path: Path
) -> Path | None:
    """Karışıklık matrisini CSV olarak yazar (son satır/sütun: arka plan)."""
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is None:
        return None
    matrix = getattr(cm_obj, "matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix)
    labels = list(class_names) + ["arka_plan"]
    if matrix.shape[0] != len(labels):
        labels = [f"c{i}" for i in range(matrix.shape[0])]
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "gercek \\ tahmin"
    df.to_csv(output_path, encoding="utf-8-sig")
    return output_path


# ==========================================================================
# Hata analizi (karar motoru seviyesinde)
# ==========================================================================
def read_ground_truth(label_path: Path, class_names: list[str]) -> set[str]:
    """Bir label dosyasındaki gerçek sınıf isimlerini döndürür."""
    if not label_path.is_file():
        return set()
    names: set[str] = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= cid < len(class_names):
            names.add(class_names[cid])
    return names


def error_analysis(
    detector: BottleDetector,
    images_dir: Path,
    class_names: list[str],
    engine: DecisionEngine,
    limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Görüntü seviyesinde yanlış pozitif ve kaçırılmış kusurları listeler."""
    labels_dir = labels_dir_for(images_dir)
    images = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit:
        images = images[:limit]

    defect_classes = {
        name for name, rule in engine.rules.items() if rule.status == STATUS_DEFECT
    }

    false_positive_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for img in images:
        truth = read_ground_truth(labels_dir / f"{img.stem}.txt", class_names)
        result, _ = detector.predict(img)
        predicted = {d.class_name for d in result.detections if d.tier == "confident"}

        truth_defects = truth & defect_classes
        pred_defects = set(result.defect_classes)

        decision_rows.append(
            {
                "image": img.name,
                "ground_truth": ";".join(sorted(truth)) or "-",
                "predicted": ";".join(sorted(predicted)) or "-",
                "decision": result.status,
                "max_confidence": round(result.max_confidence, 4),
            }
        )

        # Kaçırılmış kusur: gerçekte kusur var, karar HATALI değil (EN KRİTİK)
        if truth_defects and not result.is_defective:
            missed_rows.append(
                {
                    "image": img.name,
                    "missed_defects": ";".join(sorted(truth_defects)),
                    "decision": result.status,
                    "reason": result.reason,
                    "predicted": ";".join(sorted(predicted)) or "-",
                    "max_confidence": round(result.max_confidence, 4),
                }
            )

        # Yanlış pozitif: kusur tahmin edildi ama gerçekte yok
        spurious = pred_defects - truth
        if spurious:
            false_positive_rows.append(
                {
                    "image": img.name,
                    "false_positive_defects": ";".join(sorted(spurious)),
                    "ground_truth": ";".join(sorted(truth)) or "-",
                    "decision": result.status,
                    "max_confidence": round(result.max_confidence, 4),
                }
            )

    return {
        "false_positives": pd.DataFrame(false_positive_rows),
        "missed_defects": pd.DataFrame(missed_rows),
        "decisions": pd.DataFrame(decision_rows),
    }


# ==========================================================================
def main() -> int:
    configure_console()
    config = load_config()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(
        description="OilBottleVisionDemo - model değerlendirme"
    )
    parser.add_argument("--weights", type=Path, default=None, help="Model ağırlığı")
    parser.add_argument("--data", type=Path, default=None, help="data.yaml yolu")
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Değerlendirme bölümü (test yoksa val kullanılır)",
    )
    parser.add_argument("--device", default=None, help="'auto' | 'cpu' | '0'")
    parser.add_argument("--batch", type=int, default=8, help="Doğrulama batch size")
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        help="Görüntü bazlı hata analizini atla (daha hızlı)",
    )
    parser.add_argument(
        "--error-limit",
        type=int,
        default=None,
        help="Hata analizinde incelenecek en fazla görüntü sayısı",
    )
    args = parser.parse_args()

    data_yaml = locate_data_yaml(config, args.data)
    if data_yaml is None:
        print("HATA: data.yaml bulunamadı. Önce: python scripts/prepare_dataset.py")
        return 2

    weights = args.weights or config.trained_weights
    if not Path(weights).is_file():
        print(f"HATA: Model ağırlığı bulunamadı: {weights}")
        print("      Önce eğitim çalıştırın: python scripts/train.py")
        return 2

    # Veri setinde ayri bir test bolumu yoksa otomatik olarak val'e dus.
    available = dataset_splits_present(data_yaml)
    if args.split not in available:
        if args.split == "test" and "val" in available:
            print("  Not: veri setinde 'test' bölümü yok -> 'val' kullanılıyor.")
            args.split = "val"
        else:
            print(f"HATA: '{args.split}' bölümü veri setinde bulunamadı.")
            print(f"      Mevcut bölümler: {', '.join(available) or 'yok'}")
            return 2

    class_names = read_class_names(data_yaml)
    device = resolve_device(args.device or config.device_setting)

    print()
    print("=" * 70)
    print("  MODEL DEĞERLENDİRME")
    print("=" * 70)
    print(f"  Ağırlık  : {weights}")
    print(f"  data.yaml: {data_yaml}")
    print(f"  Bölüm    : {args.split}")
    print(f"  Cihaz    : {device}")
    print()

    # --- 1) Ultralytics val() -------------------------------------------
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    run_name = f"eval_{args.split}"
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=config.imgsz,
        batch=args.batch,
        conf=0.001,  # metrikler için standart düşük eşik
        iou=config.thresholds.iou_threshold,
        device=device,
        project=str(config.training_output_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    overall = extract_metrics(metrics)
    precision = overall.get("mp", 0.0)
    recall = overall.get("mr", 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("  GENEL METRİKLER")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    mAP50     : {overall.get('map50', 0.0):.4f}")
    print(f"    mAP50-95  : {overall.get('map', 0.0):.4f}")
    print()

    # --- 2) Sınıf bazlı ---------------------------------------------------
    per_class = per_class_table(metrics, class_names)
    per_class_path = config.reports_dir / "evaluation_per_class.csv"
    if not per_class.empty:
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")
        print("  SINIF BAZLI SONUÇLAR (kusur recall'ü en kritik metriktir)")
        width = max(len(c) for c in per_class["class_name"])
        print(
            f"    {'sınıf':<{width}}  {'precision':>9}  {'recall':>7}  "
            f"{'F1':>6}  {'mAP50':>6}  {'mAP50-95':>8}"
        )
        for _, row in per_class.iterrows():
            print(
                f"    {row['class_name']:<{width}}  {row['precision']:>9.4f}  "
                f"{row['recall']:>7.4f}  {row['f1']:>6.4f}  "
                f"{row['mAP50']:>6.4f}  {row['mAP50_95']:>8.4f}"
            )
        print()

    # En zayıf sınıflar
    weakest_recall = None
    weakest_map = None
    defect_recall: dict[str, float] = {}
    engine = DecisionEngine(config=config, class_names=class_names)
    if not per_class.empty:
        weakest_recall = per_class.loc[per_class["recall"].idxmin()].to_dict()
        weakest_map = per_class.loc[per_class["mAP50"].idxmin()].to_dict()
        for _, row in per_class.iterrows():
            rule = engine.rule_for(str(row["class_name"]))
            if rule.status == STATUS_DEFECT:
                defect_recall[str(row["class_name"])] = float(row["recall"])

        print(
            f"  EN DÜŞÜK RECALL : {weakest_recall['class_name']} "
            f"({weakest_recall['recall']:.4f})"
        )
        print(
            f"  EN DÜŞÜK mAP50  : {weakest_map['class_name']} "
            f"({weakest_map['mAP50']:.4f})"
        )
        if defect_recall:
            worst_defect = min(defect_recall, key=defect_recall.get)
            print(
                f"  EN ZAYIF KUSUR  : {worst_defect} "
                f"(recall {defect_recall[worst_defect]:.4f}) "
                "-> kaçırılan kusur riski en yüksek sınıf"
            )
        print()

    # --- 3) Confusion matrix ---------------------------------------------
    run_dir = Path(getattr(metrics, "save_dir", config.training_output_dir / run_name))
    cm_images = save_confusion_matrix(run_dir, config.reports_dir)
    cm_csv = confusion_matrix_to_csv(
        metrics, class_names, config.reports_dir / "confusion_matrix.csv"
    )
    for p in cm_images:
        print(f"  Confusion matrix: {p}")
    if cm_csv:
        print(f"  Confusion matrix (csv): {cm_csv}")
    print()

    # --- 4) Hata analizi ---------------------------------------------------
    analysis_summary: dict[str, Any] = {}
    if not args.no_error_analysis:
        raw = read_data_yaml(data_yaml)
        base = Path(str(raw.get("path"))) if raw.get("path") else data_yaml.parent
        if not base.is_absolute():
            base = data_yaml.parent / base
        images_dir = resolve_split_dir(base, raw.get(args.split), args.split)
        if images_dir is None and args.split == "test":
            images_dir = resolve_split_dir(base, raw.get("val"), "val")

        if images_dir is None:
            print("  ! Hata analizi için görüntü klasörü bulunamadı, atlanıyor.")
        else:
            print(f"  Hata analizi çalışıyor: {images_dir}")
            detector = BottleDetector(weights=weights, device=device, config=config)
            tables = error_analysis(
                detector, images_dir, class_names, detector.engine, limit=args.error_limit
            )
            fp_path = config.reports_dir / "false_positives.csv"
            md_path = config.reports_dir / "missed_defects.csv"
            dec_path = config.reports_dir / "decision_audit.csv"
            tables["false_positives"].to_csv(fp_path, index=False, encoding="utf-8-sig")
            tables["missed_defects"].to_csv(md_path, index=False, encoding="utf-8-sig")
            tables["decisions"].to_csv(dec_path, index=False, encoding="utf-8-sig")

            total = len(tables["decisions"])
            missed = len(tables["missed_defects"])
            fps = len(tables["false_positives"])
            analysis_summary = {
                "images_evaluated": total,
                "missed_defect_images": missed,
                "false_positive_images": fps,
                "missed_defect_rate": round(missed / total, 4) if total else 0.0,
                "false_positive_rate": round(fps / total, 4) if total else 0.0,
                "false_positives_csv": str(fp_path),
                "missed_defects_csv": str(md_path),
                "decision_audit_csv": str(dec_path),
            }
            print()
            print("  KARAR MOTORU HATA ANALİZİ")
            print(f"    İncelenen görüntü         : {total}")
            print(
                f"    KAÇIRILMIŞ kusur (kritik) : {missed} "
                f"(%{100 * missed / total:.1f})" if total else "    KAÇIRILMIŞ kusur: -"
            )
            print(
                f"    Yanlış pozitif            : {fps} "
                f"(%{100 * fps / total:.1f})" if total else "    Yanlış pozitif: -"
            )
            print(f"    Detay: {md_path}")
            print()

    # --- 5) JSON rapor -----------------------------------------------------
    report = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "device": device,
        "class_names": class_names,
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mAP50": round(float(overall.get("map50", 0.0)), 4),
            "mAP50_95": round(float(overall.get("map", 0.0)), 4),
        },
        "per_class": per_class.to_dict(orient="records") if not per_class.empty else [],
        "weakest_by_recall": weakest_recall,
        "weakest_by_map50": weakest_map,
        "defect_recall": defect_recall,
        "error_analysis": analysis_summary,
        "confusion_matrix_images": [str(p) for p in cm_images],
        "note": (
            "Kalite kontrolde en kritik hata, kusurlu ürünün SAĞLAM kabul "
            "edilmesidir. Kusur sınıflarının recall değerleri öncelikli olarak "
            "izlenmelidir."
        ),
    }
    report_path = config.reports_dir / "evaluation_report.json"
    write_json(report_path, report)
    print(f"  Rapor: {report_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
