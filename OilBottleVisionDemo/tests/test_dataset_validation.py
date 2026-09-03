"""Veri seti doğrulama testleri.

Testler sentetik (geçici) bir YOLO veri seti üzerinde çalışır; gerçek veri
setinin varlığına bağlı değildir. Gerçek veri seti mevcutsa ek testler
otomatik olarak devreye girer.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    detect_duplicates,
    find_data_yaml,
    find_dataset_zip,
    read_class_names,
    safe_extract_zip,
    validate_dataset,
    write_portable_data_yaml,
)

CLASS_NAMES = [
    "bottle_other_brand",
    "bottle_ron88",
    "defect_debris",
    "defect_label_damage",
    "defect_loose_cap",
    "defect_low_fill",
    "defect_no_cap",
]


# ==========================================================================
# Sentetik veri seti üretimi
# ==========================================================================
def _write_image(path: Path, seed: int) -> None:
    import cv2  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(48, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(buffer.tobytes())


def _write_label(path: Path, class_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cid} 0.5 0.5 0.3 0.4" for cid in class_ids]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_dataset(root: Path, names: list[str] | None = None) -> Path:
    """Geçerli, küçük bir YOLO veri seti oluşturur ve data.yaml yolunu döndürür."""
    names = names or CLASS_NAMES
    seed = 0
    for split, count in (("train", 4), ("valid", 2), ("test", 2)):
        for i in range(count):
            seed += 1
            _write_image(root / split / "images" / f"{split}_{i}.jpg", seed)
            _write_label(
                root / split / "labels" / f"{split}_{i}.txt", [i % len(names)]
            )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _build_dataset(tmp_path / "bottle_qa")


# ==========================================================================
# data.yaml okuma
# ==========================================================================
def test_data_yaml_siniflari_okunuyor(dataset: Path):
    """Sınıf isimleri data.yaml'dan sırayla okunmalı (tahmin edilmemeli)."""
    names = read_class_names(dataset)
    assert names == CLASS_NAMES
    assert names[0] == "bottle_other_brand"


def test_sozluk_bicimindeki_names_okunuyor(tmp_path: Path):
    """names: {0: a, 1: b} biçimi de index sırasıyla okunmalı."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"names": {1: "defect_debris", 0: "bottle_ron88"}}),
        encoding="utf-8",
    )
    assert read_class_names(data_yaml) == ["bottle_ron88", "defect_debris"]


def test_names_yoksa_hata(tmp_path: Path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"nc": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="names"):
        read_class_names(data_yaml)


def test_data_yaml_bulunuyor(dataset: Path):
    found = find_data_yaml(dataset.parent)
    assert found is not None
    assert found.name == "data.yaml"


# ==========================================================================
# Bölüm doğrulama
# ==========================================================================
def test_gecerli_veri_seti_dogrulaniyor(dataset: Path):
    report = validate_dataset(dataset)
    assert report.is_valid, report.errors
    assert report.class_names == CLASS_NAMES
    assert report.splits["train"].image_count == 4
    assert report.splits["val"].image_count == 2
    assert report.splits["test"].image_count == 2
    assert report.total_images == 8


def test_valid_klasoru_val_olarak_taniniyor(dataset: Path):
    """'valid' klasör adı 'val' bölümü olarak çözülmeli."""
    report = validate_dataset(dataset)
    assert report.splits["val"].images_dir is not None
    assert report.splits["val"].images_dir.parent.name == "valid"


def test_eksik_train_bolumu_hata_veriyor(dataset: Path):
    import shutil  # noqa: PLC0415

    shutil.rmtree(dataset.parent / "train")
    report = validate_dataset(dataset)
    assert not report.is_valid
    assert any("train" in e for e in report.errors)


# ==========================================================================
# Görüntü / label eşleşmesi
# ==========================================================================
def test_eksik_label_tespit_ediliyor(dataset: Path):
    """Label dosyası olmayan görüntü raporlanmalı."""
    (dataset.parent / "train" / "labels" / "train_0.txt").unlink()
    report = validate_dataset(dataset)
    assert "train_0.jpg" in report.splits["train"].missing_labels
    assert any("label dosyası yok" in w for w in report.warnings)


def test_karsiligi_olmayan_label_tespit_ediliyor(dataset: Path):
    """Görüntüsü olmayan label dosyası raporlanmalı."""
    _write_label(dataset.parent / "train" / "labels" / "hayalet.txt", [0])
    report = validate_dataset(dataset)
    assert "hayalet.txt" in report.splits["train"].orphan_labels


def test_bos_label_tespit_ediliyor(dataset: Path):
    (dataset.parent / "train" / "labels" / "train_1.txt").write_text("", encoding="utf-8")
    report = validate_dataset(dataset)
    assert "train_1.txt" in report.splits["train"].empty_labels


def test_bozuk_label_tespit_ediliyor(dataset: Path):
    """Eksik alanlı label dosyası hata olarak raporlanmalı."""
    (dataset.parent / "train" / "labels" / "train_2.txt").write_text(
        "0 0.5 0.5\n", encoding="utf-8"
    )
    report = validate_dataset(dataset)
    assert not report.is_valid
    assert report.splits["train"].corrupt_labels


def test_gecersiz_sinif_id_tespit_ediliyor(dataset: Path):
    """data.yaml'daki sınıf sayısının dışındaki id yakalanmalı."""
    _write_label(dataset.parent / "train" / "labels" / "train_3.txt", [99])
    report = validate_dataset(dataset)
    assert not report.is_valid
    assert report.splits["train"].out_of_range_class_ids


def test_normalize_edilmemis_koordinat_tespit_ediliyor(dataset: Path):
    (dataset.parent / "train" / "labels" / "train_0.txt").write_text(
        "0 250 300 40 80\n", encoding="utf-8"
    )
    report = validate_dataset(dataset)
    assert not report.is_valid


def test_bozuk_goruntu_tespit_ediliyor(dataset: Path):
    (dataset.parent / "train" / "images" / "train_0.jpg").write_bytes(b"bu bir jpeg degil")
    report = validate_dataset(dataset)
    assert not report.is_valid
    assert "train_0.jpg" in report.splits["train"].corrupt_images


# ==========================================================================
# Sınıf dağılımı
# ==========================================================================
def test_sinif_dagilimi_hesaplaniyor(dataset: Path):
    report = validate_dataset(dataset)
    totals = report.total_class_box_counts()
    assert sum(totals.values()) == 8  # her görüntüde 1 kutu
    assert totals["bottle_other_brand"] >= 1


def test_dagilim_dataframe_uretiliyor(dataset: Path):
    from src.dataset import distribution_dataframe  # noqa: PLC0415

    df = distribution_dataframe(validate_dataset(dataset))
    assert list(df["class_name"]) == CLASS_NAMES
    assert df["total_boxes"].sum() == 8


# ==========================================================================
# Kopya / veri sızıntısı
# ==========================================================================
def test_bolumler_arasi_kopya_tespit_ediliyor(dataset: Path):
    """Aynı görüntü train ve test'te varsa sızıntı olarak raporlanmalı."""
    import shutil  # noqa: PLC0415

    source = dataset.parent / "train" / "images" / "train_0.jpg"
    shutil.copy2(source, dataset.parent / "test" / "images" / "sizinti.jpg")
    _write_label(dataset.parent / "test" / "labels" / "sizinti.txt", [0])

    report = validate_dataset(dataset)
    assert report.duplicates_across_splits
    leak = report.duplicates_across_splits[0]
    assert set(leak["splits"]) == {"train", "test"}
    assert any("veri sızıntısı" in w for w in report.warnings)


def test_kopya_yoksa_bos_liste(dataset: Path):
    report = validate_dataset(dataset)
    assert report.duplicates_across_splits == []


def test_detect_duplicates_dogrudan_calisiyor(dataset: Path):
    report = validate_dataset(dataset, check_duplicates=False)
    across, within = detect_duplicates(report.splits)
    assert across == []
    assert within == []


# ==========================================================================
# Taşınabilir data.yaml
# ==========================================================================
def test_tasinabilir_data_yaml_yollari_duzeltiyor(dataset: Path, tmp_path: Path):
    """Mutlak/bozuk yollar proje içi göreli yollara çevrilmeli."""
    dataset.write_text(
        yaml.safe_dump(
            {
                "train": "C:/eski/makine/train/images",
                "val": "/mnt/eski/valid/images",
                "test": "../test/images",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            }
        ),
        encoding="utf-8",
    )
    report = validate_dataset(dataset)
    out = write_portable_data_yaml(report, dataset.parent / "data_portable.yaml")

    assert out.is_file()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["names"] == CLASS_NAMES
    assert data["nc"] == len(CLASS_NAMES)
    # Yollar artık göreli ve mevcut olmalı
    base = Path(data["path"])
    for split in ("train", "val", "test"):
        assert not Path(data[split]).is_absolute()
        assert (base / data[split]).is_dir()


def test_kaynak_data_yaml_degistirilmiyor(dataset: Path):
    """Hazırlama işlemi kaynak data.yaml'ı bozmamalı."""
    before = dataset.read_text(encoding="utf-8")
    report = validate_dataset(dataset)
    write_portable_data_yaml(report, dataset.parent / "data_portable.yaml")
    assert dataset.read_text(encoding="utf-8") == before


# ==========================================================================
# ZIP işlemleri
# ==========================================================================
def test_zip_guvenli_cikariliyor(dataset: Path, tmp_path: Path):
    zip_path = tmp_path / "bottle_qa.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in dataset.parent.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(dataset.parent))

    dest = tmp_path / "cikarilan"
    safe_extract_zip(zip_path, dest)
    assert (dest / "data.yaml").is_file()
    assert read_class_names(dest / "data.yaml") == CLASS_NAMES


def test_zip_slip_engelleniyor(tmp_path: Path):
    """Hedef klasörün dışına yazmaya çalışan ZIP reddedilmeli."""
    zip_path = tmp_path / "kotu.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../kacis.txt", "zararli")

    with pytest.raises(ValueError, match="Güvenli olmayan"):
        safe_extract_zip(zip_path, tmp_path / "hedef")
    assert not (tmp_path / "kacis.txt").exists()


def test_zip_arama_tercih_sirasi(tmp_path: Path, dataset: Path):
    """bottle_qa.zip diğer ZIP dosyalarına tercih edilmeli."""
    (tmp_path / "aaa_baska.zip").write_bytes(b"PK\x03\x04bozuk")
    zip_path = tmp_path / "bottle_qa.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(dataset, "data.yaml")

    found = find_dataset_zip([tmp_path])
    assert found == zip_path


def test_zip_yoksa_none(tmp_path: Path):
    assert find_dataset_zip([tmp_path / "olmayan"]) is None


# ==========================================================================
# Gerçek veri seti (varsa)
# ==========================================================================
def test_gercek_veri_seti_siniflari():
    """Hazırlanmış gerçek veri seti varsa sınıfları data.yaml'dan okunmalı."""
    config = load_config()
    data_yaml = find_data_yaml(config.data_prepared_dir)
    if data_yaml is None:
        pytest.skip("Hazırlanmış veri seti yok (prepare_dataset.py çalıştırılmamış).")
    names = read_class_names(data_yaml)
    assert names, "data.yaml içinde sınıf bulunamadı"
    assert len(names) == len(set(names)), "Yinelenen sınıf adı var"
