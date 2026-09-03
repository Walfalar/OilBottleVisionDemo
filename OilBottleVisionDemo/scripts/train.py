"""YOLO eğitimi.

Model seçimi modülerdir: varsayılan ``yolo26n.pt``; kurulu Ultralytics sürümü
desteklemiyorsa otomatik olarak ``yolo11n.pt`` ile devam edilir.

Kullanım:
    python scripts/train.py --smoke                 # 3 epoch hızlı test
    python scripts/train.py                         # 50 epoch ana eğitim
    python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    find_data_yaml,
    find_extracted_dataset,
    read_class_names,
)
from src.utils import (  # noqa: E402
    configure_console,
    file_timestamp,
    hardware_info,
    human_duration,
    resolve_device,
    suggest_batch_size,
    write_json,
)

PREPARED_NAME = "bottle_qa"


def locate_data_yaml(config, explicit: Path | None) -> Path | None:
    """Eğitimde kullanılacak data.yaml'ı bulur (taşınabilir kopya öncelikli)."""
    if explicit is not None:
        return Path(explicit) if Path(explicit).is_file() else None

    prepared_root = config.data_prepared_dir / PREPARED_NAME
    portable = prepared_root / "data_portable.yaml"
    if portable.is_file():
        return portable
    found = find_data_yaml(config.data_prepared_dir)
    if found is not None:
        return found
    # Hazırlanmış kopya yoksa doğrudan kaynak klasördeki veri setini kullan.
    return find_extracted_dataset([config.data_source_dir, config.root / "data"])


def build_train_kwargs(args, config, data_yaml: Path, device: str) -> dict[str, Any]:
    """Ultralytics train() argümanlarını yapılandırmadan üretir."""
    training = config.training
    aug = config.augment

    epochs = args.epochs
    if epochs is None:
        epochs = int(training.get("smoke_epochs", 3)) if args.smoke else int(
            training.get("epochs", 50)
        )

    batch = args.batch
    if batch is None:
        configured = training.get("batch", -1)
        # -1 => Ultralytics otomatik VRAM tahsisi; CPU'da desteklenmez.
        if device == "cpu":
            batch = suggest_batch_size(device, args.imgsz or config.imgsz)
        else:
            batch = configured if configured is not None else -1

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(args.imgsz or config.imgsz),
        "batch": batch,
        "device": device,
        "patience": int(args.patience or training.get("patience", 10)),
        "seed": int(args.seed or training.get("seed", 42)),
        "workers": int(training.get("workers", 4)),
        "optimizer": str(training.get("optimizer", "auto")),
        "cache": bool(training.get("cache", False)),
        "project": str(config.training_output_dir),
        "name": args.run_name,
        "exist_ok": True,
        "verbose": True,
        "plots": True,
        "val": True,
    }

    # Kalite kontrolde renk/konum hassasiyeti önemli: augmentation ölçülü.
    for key in (
        "flipud",
        "fliplr",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "close_mosaic",
        "mixup",
        "copy_paste",
        "erasing",
    ):
        if key in aug:
            value = aug[key]
            kwargs[key] = int(value) if key == "close_mosaic" else float(value)

    # Smoke testte mosaic tamamen kapalı olsun (kısa eğitimde gürültü yapar).
    if args.smoke:
        kwargs["mosaic"] = 0.0
        kwargs["close_mosaic"] = 0

    return kwargs


def copy_weights(run_dir: Path, config) -> dict[str, str | None]:
    """best.pt / last.pt dosyalarını models/ altına kopyalar."""
    config.models_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str | None] = {"best": None, "last": None}
    for kind, target in (("best", config.trained_weights), ("last", config.last_weights)):
        source = run_dir / "weights" / f"{kind}.pt"
        if source.is_file():
            shutil.copy2(source, target)
            result[kind] = str(target)
    return result


def extract_metrics(metrics: Any) -> dict[str, Any]:
    """Ultralytics metriklerini JSON'a yazılabilir sözlüğe çevirir."""
    output: dict[str, Any] = {}
    if metrics is None:
        return output

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                output[str(key)] = float(value)
            except (TypeError, ValueError):
                output[str(key)] = str(value)

    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("mp", "mr", "map50", "map75", "map"):
            value = getattr(box, attr, None)
            if value is not None:
                try:
                    output[attr] = float(value)
                except (TypeError, ValueError):
                    pass
    return output


def find_best_epoch(run_dir: Path, model: Any, total_epochs: int) -> int | None:
    """En iyi epoch numarasını (1 tabanlı) belirler.

    Öncelik results.csv üzerinden Ultralytics fitness hesabıdır
    (0.1 * mAP50 + 0.9 * mAP50-95); bu, sürümden bağımsız ve doğrulanabilirdir.
    Dosya okunamazsa trainer özniteliğine düşülür ve geçerli aralığa kırpılır.
    """
    results_csv = Path(run_dir) / "results.csv"
    if results_csv.is_file():
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            map50 = df.get("metrics/mAP50(B)")
            map5095 = df.get("metrics/mAP50-95(B)")
            if map5095 is not None and not map5095.empty:
                fitness = 0.9 * map5095 + (0.1 * map50 if map50 is not None else 0.0)
                row = df.loc[fitness.idxmax()]
                epoch = row.get("epoch")
                if epoch is not None:
                    # Ultralytics 'epoch' sütunu 1 tabanlıdır.
                    return int(epoch)
        except (OSError, ValueError, KeyError):
            pass

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return None
    raw = getattr(trainer, "best_epoch", None)
    if raw is None:
        raw = getattr(getattr(trainer, "stopper", None), "best_epoch", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    # Sürüme göre 0 ya da 1 tabanlı olabiliyor; geçerli aralığa kırp.
    return max(1, min(value if value >= 1 else value + 1, int(total_epochs)))


def main() -> int:
    configure_console()
    config = load_config()
    config.ensure_dirs()
    training_defaults = config.training

    parser = argparse.ArgumentParser(description="OilBottleVisionDemo - YOLO eğitimi")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model ağırlığı (varsayılan: {config.model_weights})",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch sayısı")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Hızlı doğrulama ({training_defaults.get('smoke_epochs', 3)} epoch)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (verilmezse GPU belleğine göre otomatik)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Görüntü boyutu")
    parser.add_argument(
        "--device", default=None, help="'auto' | 'cpu' | '0' (varsayılan: auto)"
    )
    parser.add_argument("--patience", type=int, default=None, help="Early stopping")
    parser.add_argument("--seed", type=int, default=None, help="Rastgelelik tohumu")
    parser.add_argument("--data", type=Path, default=None, help="data.yaml yolu")
    parser.add_argument("--run-name", default=None, help="Çalışma klasörü adı")
    args = parser.parse_args()

    if args.run_name is None:
        prefix = "smoke" if args.smoke else "train"
        args.run_name = f"{prefix}_{file_timestamp()}"

    # --- veri seti ------------------------------------------------------
    data_yaml = locate_data_yaml(config, args.data)
    if data_yaml is None:
        print("HATA: Eğitim için data.yaml bulunamadı.")
        print("      Önce şu komutu çalıştırın: python scripts/prepare_dataset.py")
        return 2

    try:
        class_names = read_class_names(data_yaml)
    except (ValueError, OSError) as exc:
        print(f"HATA: data.yaml okunamadı: {exc}")
        return 1

    # --- donanım ---------------------------------------------------------
    device = resolve_device(args.device or config.device_setting)
    hw = hardware_info()

    print()
    print("=" * 70)
    print("  EĞİTİM BAŞLIYOR")
    print("=" * 70)
    print(f"  data.yaml : {data_yaml}")
    print(f"  Sınıflar  : {len(class_names)} -> {', '.join(class_names)}")
    print(f"  Cihaz     : {device}" + ("" if device != "cpu" else "  (GPU bulunamadı)"))
    for gpu in hw["gpus"]:
        print(f"  GPU       : {gpu['name']} ({gpu['total_memory_gb']} GB)")
    print()

    # --- model yükleme (fallback ile) -------------------------------------
    from ultralytics import YOLO  # noqa: PLC0415

    import ultralytics  # noqa: PLC0415

    requested = args.model or config.model_weights
    fallback = config.fallback_weights
    candidates = [requested] if requested == fallback else [requested, fallback]

    model = None
    used_weights = None
    load_errors: list[str] = []
    for candidate in candidates:
        try:
            model = YOLO(candidate)
            used_weights = candidate
            break
        except Exception as exc:  # noqa: BLE001 - ultralytics çeşitli hata tipleri atar
            load_errors.append(f"{candidate}: {exc}")
            print(f"  ! '{candidate}' yüklenemedi: {exc}")
            if candidate != candidates[-1]:
                print(f"  -> Yedek modele geçiliyor: {candidates[-1]}")

    if model is None:
        print("HATA: Hiçbir model ağırlığı yüklenemedi:")
        for err in load_errors:
            print(f"  - {err}")
        return 1

    print(f"  Ultralytics: {ultralytics.__version__}")
    print(f"  Model      : {used_weights}"
          + ("  (yedek model kullanıldı)" if used_weights != requested else ""))

    kwargs = build_train_kwargs(args, config, data_yaml, device)
    print(f"  Epoch      : {kwargs['epochs']}")
    print(f"  Batch      : {kwargs['batch']}" + (" (otomatik)" if kwargs["batch"] == -1 else ""))
    print(f"  imgsz      : {kwargs['imgsz']}")
    print(f"  Patience   : {kwargs['patience']}   Seed: {kwargs['seed']}")
    print(f"  Çıktı      : {config.training_output_dir / args.run_name}")
    print()

    # --- eğitim ------------------------------------------------------------
    start = time.time()
    try:
        results = model.train(**kwargs)
    except Exception as exc:  # noqa: BLE001 - eğitim çeşitli hatalar verebilir
        print(f"HATA: Eğitim başarısız: {exc}")
        return 1
    duration = time.time() - start

    save_dir = getattr(results, "save_dir", None)
    run_dir = Path(save_dir) if save_dir else (config.training_output_dir / args.run_name)
    if not run_dir.is_dir():
        run_dir = config.training_output_dir / args.run_name

    weights = copy_weights(run_dir, config)

    # --- metrikler ---------------------------------------------------------
    metrics = extract_metrics(getattr(model, "metrics", None))
    best_epoch = find_best_epoch(run_dir, model, kwargs["epochs"])

    report = {
        "run_name": args.run_name,
        "mode": "smoke" if args.smoke else "full",
        "requested_model": requested,
        "used_model": used_weights,
        "fallback_used": used_weights != requested,
        "ultralytics_version": ultralytics.__version__,
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "device": device,
        "hardware": hw,
        "epochs": kwargs["epochs"],
        "best_epoch": best_epoch,
        "batch": kwargs["batch"],
        "imgsz": kwargs["imgsz"],
        "seed": kwargs["seed"],
        "patience": kwargs["patience"],
        "augmentation": {k: kwargs[k] for k in config.augment if k in kwargs},
        "duration_seconds": round(duration, 1),
        "duration_human": human_duration(duration),
        "run_dir": str(run_dir),
        "weights": weights,
        "metrics": metrics,
    }
    report_path = run_dir / "training_report.json"
    write_json(report_path, report)
    write_json(
        config.reports_dir / f"training_report_{args.run_name}.json", report
    )

    print()
    print("=" * 70)
    print("  EĞİTİM TAMAMLANDI")
    print("=" * 70)
    print(f"  Süre        : {human_duration(duration)}")
    print(f"  En iyi epoch: {best_epoch if best_epoch is not None else '-'}")
    print(f"  best.pt     : {weights['best'] or 'oluşturulamadı'}")
    print(f"  last.pt     : {weights['last'] or 'oluşturulamadı'}")
    if metrics:
        print("  Metrikler:")
        for key in ("mp", "mr", "map50", "map"):
            if key in metrics:
                pretty = {
                    "mp": "Precision",
                    "mr": "Recall",
                    "map50": "mAP50",
                    "map": "mAP50-95",
                }[key]
                print(f"    {pretty:<10}: {metrics[key]:.4f}")
    print(f"  Rapor       : {report_path}")
    print()
    print("  Sıradaki adım:")
    print("    python scripts/evaluate.py")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
