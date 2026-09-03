"""Veri setini güvenli biçimde çıkarır, doğrular ve taşınabilir hâle getirir.

Kaynak ZIP / veri seti DEĞİŞTİRİLMEZ. Düzeltilmiş kopya
``data/prepared/bottle_qa`` altında oluşturulur.

Kullanım:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --zip data/source/bottle_qa.zip --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    KAGGLE_DATASET_HANDLE,
    dataset_splits_present,
    download_kaggle_dataset,
    find_data_yaml,
    find_dataset_zip,
    find_extracted_dataset,
    safe_extract_zip,
    validate_dataset,
    write_distribution_reports,
    write_portable_data_yaml,
)
from src.decision_engine import DecisionEngine  # noqa: E402
from src.utils import configure_console, write_json  # noqa: E402
from scripts.inspect_dataset import (  # noqa: E402
    DATASET_URL,
    dataset_search_roots,
    print_missing_dataset_help,
    print_validation,
    search_roots,
)

PREPARED_NAME = "bottle_qa"


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description="BOTTLE_QA veri setini hazırlar ve doğrular."
    )
    parser.add_argument("--zip", type=Path, default=None, help="Kaynak ZIP yolu")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Çıkarılmış veri setinin data.yaml yolu (otomatik aramayı atlar)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="data/prepared altındaki mevcut kopyayı silip yeniden çıkar",
    )
    parser.add_argument(
        "--no-image-check", action="store_true", help="Bozuk görüntü kontrolünü atla"
    )
    parser.add_argument(
        "--no-duplicate-check",
        action="store_true",
        help="Hash tabanlı kopya kontrolünü atla",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Veri seti yoksa kagglehub ile indir ({KAGGLE_DATASET_HANDLE})",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Veri seti bulunamazsa indirmeyi hiç deneme",
    )
    args = parser.parse_args()

    config = load_config()
    config.ensure_dirs()

    print()
    print("=" * 70)
    print("  VERİ SETİ HAZIRLAMA")
    print("=" * 70)
    print(f"  Kaynak veri seti: {DATASET_URL}")
    print()

    prepared_root = config.data_prepared_dir / PREPARED_NAME

    # --- 1) Veri setini bul ------------------------------------------
    # Öncelik: --data-yaml > mevcut çıkarılmış veri seti (data/prepared,
    # data/source) > ZIP > kagglehub indirme. Veri seti mevcutsa ASLA
    # yeniden indirilmez/çıkarılmaz.
    if args.force and prepared_root.exists():
        print(f"  --force: mevcut kopya siliniyor -> {prepared_root}")
        shutil.rmtree(prepared_root)

    existing_yaml: Path | None = None
    if args.data_yaml is not None:
        existing_yaml = Path(args.data_yaml)
        if not existing_yaml.is_file():
            print(f"  HATA: data.yaml bulunamadı: {existing_yaml}")
            return 2
        print(f"  Verilen data.yaml kullanılıyor: {existing_yaml}")
    elif not args.force and not args.download:
        existing_yaml = find_extracted_dataset(dataset_search_roots(config))
        if existing_yaml is not None:
            print("  Mevcut veri seti bulundu (indirme/çıkarma atlanıyor).")
            print(f"  Kök klasör  : {existing_yaml.parent}")

    # ZIP'ten çıkarma
    if existing_yaml is None and not args.download:
        zip_path = args.zip or find_dataset_zip(search_roots(config))
        if zip_path is not None and Path(zip_path).is_file():
            print(f"  ZIP bulundu : {zip_path}")
            print(f"  Çıkarılıyor -> {prepared_root}")
            try:
                safe_extract_zip(Path(zip_path), prepared_root)
            except (ValueError, OSError) as exc:
                print(f"  HATA: ZIP çıkarılamadı: {exc}")
                return 1
            existing_yaml = find_data_yaml(prepared_root)
            if existing_yaml is None:
                print("  HATA: Çıkarılan içerikte data.yaml bulunamadı.")
                print("        ZIP'in YOLO formatında olduğundan emin olun.")
                return 1
            print("  Çıkarma tamamlandı.")

    # Kaggle'dan indirme (yalnızca hiçbir yerde bulunamadıysa)
    if existing_yaml is None and not args.no_download:
        print(f"  Veri seti yerelde yok. Kaggle'dan indiriliyor: {KAGGLE_DATASET_HANDLE}")
        try:
            existing_yaml = download_kaggle_dataset(
                KAGGLE_DATASET_HANDLE, dest_dir=config.data_source_dir
            )
        except RuntimeError as exc:
            print("  HATA: Kaggle indirmesi başarısız:")
            print(f"    {exc}")
            existing_yaml = None
        else:
            print(f"  İndirme tamamlandı -> {existing_yaml.parent}")

    if existing_yaml is None:
        print_missing_dataset_help(config)
        return 2

    existing_yaml = Path(existing_yaml)
    print(f"  data.yaml   : {existing_yaml}")
    splits = dataset_splits_present(existing_yaml)
    for split in ("train", "val", "test"):
        found = splits.get(split)
        print(f"  {split:<11} : {found if found else 'yok'}")
    print()

    # --- 2) Doğrulama --------------------------------------------------
    print("  Doğrulama çalışıyor (görüntü/label eşleşmesi, kopya kontrolü)...")
    report = validate_dataset(
        Path(existing_yaml),
        check_images=not args.no_image_check,
        check_duplicates=not args.no_duplicate_check,
    )
    print_validation(report)

    # --- 3) Karar motoru eşleme kontrolü -------------------------------
    engine = DecisionEngine(config=config, class_names=report.class_names)
    unmapped = engine.unmapped_classes(report.class_names)
    if unmapped:
        print("  ! configs/project.yaml -> class_mapping içinde karşılığı olmayan")
        print("    sınıflar (bunlar MANUEL KONTROL olarak değerlendirilir):")
        for name in unmapped:
            print(f"      - {name}")
        print()

    # --- 4) Taşınabilir data.yaml --------------------------------------
    prepared_root.mkdir(parents=True, exist_ok=True)
    portable = write_portable_data_yaml(
        report, prepared_root / "data_portable.yaml"
    )
    print(f"  Taşınabilir data.yaml yazıldı: {portable}")

    # --- 5) Raporlar ----------------------------------------------------
    outputs = write_distribution_reports(report, config.reports_dir)
    for key, path in outputs.items():
        print(f"  Rapor ({key}): {path}")

    json_path = config.reports_dir / "dataset_validation.json"
    payload = report.as_dict()
    payload["portable_data_yaml"] = str(portable)
    payload["unmapped_classes"] = unmapped
    write_json(json_path, payload)
    print(f"  Rapor (json): {json_path}")
    print()

    if not report.is_valid:
        print("  SONUÇ: Veri setinde düzeltilmesi gereken hatalar var (yukarıya bakın).")
        return 1

    print("  SONUÇ: Veri seti eğitime hazır.")
    print()
    print("  Sıradaki adım (3 epoch smoke test):")
    print("    python scripts/train.py --smoke")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
