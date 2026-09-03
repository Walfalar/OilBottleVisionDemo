"""Veri setini bulur ve mevcut durumunu raporlar (hiçbir şeyi değiştirmez).

Kullanım:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --zip C:/yol/bottle_qa.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    DATASET_URL,
    KAGGLE_DATASET_HANDLE,
    dataset_splits_present,
    find_dataset_zip,
    find_extracted_dataset,
    read_class_names,
    validate_dataset,
)
from src.utils import configure_console, hardware_info  # noqa: E402


def search_roots(config) -> list[Path]:
    """ZIP dosyasının aranacağı konumlar (öncelik sırasıyla)."""
    return [
        config.data_source_dir,
        config.root,
        config.root / "data",
        config.root.parent,
    ]


def dataset_search_roots(config) -> list[Path]:
    """Çıkarılmış veri setinin aranacağı konumlar (öncelik sırasıyla)."""
    return [
        config.data_prepared_dir,
        config.data_source_dir,
        config.root / "data",
    ]


def print_hardware() -> None:
    info = hardware_info()
    print("=" * 70)
    print("DONANIM VE ORTAM")
    print("=" * 70)
    print(f"  Platform      : {info['platform']}")
    print(f"  Python        : {info['python']}")
    print(f"  PyTorch       : {info['torch'] or 'kurulu değil'}")
    print(f"  CUDA (torch)  : {info['cuda_version'] or '-'}")
    print(f"  CUDA kullanımı: {'EVET' if info['cuda_available'] else 'HAYIR (CPU)'}")
    for gpu in info["gpus"]:
        print(
            f"  GPU {gpu['index']}         : {gpu['name']} "
            f"({gpu['total_memory_gb']} GB, compute {gpu['capability']})"
        )
    print()


def print_missing_dataset_help(config) -> None:
    target = config.data_source_dir / "bottle_qa.zip"
    print("!" * 70)
    print("VERİ SETİ BULUNAMADI")
    print("!" * 70)
    print()
    print("Seçenek A - otomatik indirme (kagglehub):")
    print("       pip install kagglehub")
    print("       python scripts/prepare_dataset.py --download")
    print(f"     Kaggle handle: {KAGGLE_DATASET_HANDLE}")
    print()
    print("Seçenek B - elle indirme:")
    print(f"  1) Veri setini indirin: {DATASET_URL}")
    print("     Format: YOLO (images/ + labels/ + data.yaml)")
    print("  2) ZIP'i şu konuma koyun ...")
    print(f"       {target}")
    print(f"     ... veya içeriği doğrudan şuraya çıkarın: {config.data_source_dir}")
    print()
    print("  3) Ardından hazırlama komutunu çalıştırın:")
    print("       python scripts/prepare_dataset.py")
    print()
    print("Aranan konumlar:")
    for root in dataset_search_roots(config):
        print(f"  - {root}")
    print()


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description="BOTTLE_QA veri setini bulur ve doğrular (salt okunur)."
    )
    parser.add_argument("--zip", type=Path, default=None, help="ZIP dosyasının yolu")
    parser.add_argument(
        "--data-yaml", type=Path, default=None, help="Çıkarılmış data.yaml yolu"
    )
    parser.add_argument(
        "--no-image-check",
        action="store_true",
        help="Bozuk görüntü kontrolünü atla (daha hızlı)",
    )
    parser.add_argument(
        "--no-duplicate-check",
        action="store_true",
        help="Hash tabanlı kopya kontrolünü atla (daha hızlı)",
    )
    args = parser.parse_args()

    config = load_config()
    config.ensure_dirs()

    print()
    print("=" * 70)
    print(f"  {config.raw.get('project', {}).get('name', 'OilBottleVisionDemo')} "
          "- VERİ SETİ İNCELEMESİ")
    print("=" * 70)
    print(f"Proje kökü: {config.root}")
    print()

    print_hardware()

    # 1) Çıkarılmış veri seti var mı? (önce data/prepared, sonra data/source)
    data_yaml = args.data_yaml
    if data_yaml is None:
        data_yaml = find_extracted_dataset(dataset_search_roots(config))

    if data_yaml is not None and Path(data_yaml).is_file():
        print("=" * 70)
        print("ÇIKARILMIŞ VERİ SETİ")
        print("=" * 70)
        print(f"  data.yaml: {data_yaml}")
        splits = dataset_splits_present(Path(data_yaml))
        for split in ("train", "val", "test"):
            found = splits.get(split)
            print(f"  {split:<5}   : {found if found else 'yok'}")
        try:
            names = read_class_names(Path(data_yaml))
            print(f"  Sınıflar ({len(names)}): {', '.join(names)}")
        except ValueError as exc:
            print(f"  ! Sınıflar okunamadı: {exc}")
        print()
        print("Doğrulama çalışıyor...")
        report = validate_dataset(
            Path(data_yaml),
            check_images=not args.no_image_check,
            check_duplicates=not args.no_duplicate_check,
        )
        print_validation(report)
        return 0 if report.is_valid else 1

    # 2) ZIP var mı?
    zip_path = args.zip or find_dataset_zip(search_roots(config))
    print("=" * 70)
    print("KAYNAK ZIP ARAMASI")
    print("=" * 70)
    if zip_path and Path(zip_path).is_file():
        size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
        print(f"  BULUNDU: {zip_path}  ({size_mb:.1f} MB)")
        print()
        print("Sıradaki adım:")
        print("  python scripts/prepare_dataset.py")
        print()
        return 0

    print("  ZIP bulunamadı.")
    print()
    print_missing_dataset_help(config)
    return 2


def print_validation(report) -> None:
    print()
    print("=" * 70)
    print("DOĞRULAMA SONUCU")
    print("=" * 70)
    print(f"  Kök klasör : {report.dataset_root}")
    print(f"  Sınıf sayısı: {len(report.class_names)}")
    print(f"  Toplam görüntü: {report.total_images}")
    print()
    print("  Bölümler:")
    for split, sr in report.splits.items():
        if sr.images_dir is None:
            print(f"    - {split:<6}: YOK")
            continue
        print(
            f"    - {split:<6}: {sr.image_count:>5} görüntü, "
            f"{sr.label_count:>5} label, "
            f"eksik label={len(sr.missing_labels)}, "
            f"boş label={len(sr.empty_labels)}, "
            f"bozuk={len(sr.corrupt_labels) + len(sr.corrupt_images)}"
        )
    print()
    totals = report.total_class_box_counts()
    if totals:
        print("  Sınıf başına annotation:")
        width = max(len(c) for c in report.class_names) if report.class_names else 20
        for name in report.class_names:
            print(f"    {name:<{width}} : {totals.get(name, 0)}")
        print()

    if report.duplicates_across_splits:
        print(
            f"  ! {len(report.duplicates_across_splits)} görüntü birden fazla "
            "bölümde (veri sızıntısı riski)"
        )
    if report.warnings:
        print("  UYARILAR:")
        for w in report.warnings:
            print(f"    - {w}")
    if report.errors:
        print("  HATALAR:")
        for e in report.errors:
            print(f"    - {e}")
    print()
    print(f"  SONUÇ: {'GEÇERLİ' if report.is_valid else 'GEÇERSİZ'}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
