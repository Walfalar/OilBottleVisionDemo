"""Veri seti keşfi, güvenli çıkarma, doğrulama ve raporlama.

Sınıf isimleri ASLA tahmin edilmez; her zaman veri setinin ``data.yaml``
dosyasından okunur.
"""

from __future__ import annotations

import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .utils import IMAGE_SUFFIXES, file_hash

# YOLO veri setlerinde beklenen bölümler
SPLITS: tuple[str, ...] = ("train", "val", "test")
# data.yaml içinde val için kullanılabilen alternatif anahtarlar
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}


# Veri seti kaynağı: Kaggle (Roboflow bağlantısı indirme hatası verdiği için
# artık kullanılmıyor). Sınıf isim/indexleri HER ZAMAN data.yaml'dan okunur.
KAGGLE_DATASET_HANDLE = "jihadakbr/water-bottle-defect-level-detection-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_HANDLE}"


# ==========================================================================
# Keşif
# ==========================================================================
def find_dataset_zip(search_roots: Iterable[Path]) -> Path | None:
    """Veri seti ZIP dosyasını bilinen konumlarda arar.

    Öncelik sırası: data/source/bottle_qa.zip -> dataset.zip -> bottle_qa.zip
    -> proje içindeki diğer ZIP dosyaları.
    """
    preferred_names = ("bottle_qa.zip", "dataset.zip", "bottle-qa.zip", "bottleqa.zip")
    roots = [Path(r) for r in search_roots if Path(r).exists()]

    for root in roots:
        for name in preferred_names:
            candidate = root / name
            if candidate.is_file():
                return candidate

    # İsimden bağımsız: içinde data.yaml barındıran ilk ZIP.
    for root in roots:
        for candidate in sorted(root.glob("*.zip")):
            if _zip_contains_data_yaml(candidate):
                return candidate
    return None


def _zip_contains_data_yaml(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return any(Path(n).name == "data.yaml" for n in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def find_data_yaml(root: Path) -> Path | None:
    """Klasör ağacında ``data.yaml`` dosyasını bulur (en sığ olan önce)."""
    root = Path(root)
    if not root.exists():
        return None
    if root.is_file() and root.name in ("data.yaml", "data.yml"):
        return root
    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name in ("data.yaml", "data.yml")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.relative_to(root).parts), str(p)))[0]


def find_dataset_root(search_roots: Iterable[Path]) -> Path | None:
    """Çıkarılmış bir veri setinin kök klasörünü (data.yaml'ın bulunduğu) döndürür."""
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        data_yaml = find_data_yaml(root)
        if data_yaml is not None:
            return data_yaml.parent
    return None


def dataset_splits_present(data_yaml: Path) -> dict[str, Path]:
    """``data.yaml``'a göre DİSKTE gerçekten bulunan split klasörlerini döndürür.

    train/valid/test klasörleri otomatik bulunur: hem ``images/train`` hem
    ``train/images`` düzeni, hem de ``val``/``valid``/``validation`` adları
    desteklenir. Yalnızca içinde en az bir görüntü olan klasörler sayılır.
    """
    data_yaml = Path(data_yaml)
    try:
        raw = read_data_yaml(data_yaml)
    except (OSError, yaml.YAMLError, ValueError):
        return {}

    dataset_root = data_yaml.parent
    base = dataset_root
    if raw.get("path"):
        p = Path(str(raw["path"]))
        candidate = p if p.is_absolute() else (dataset_root / p)
        if candidate.is_dir():
            base = candidate

    found: dict[str, Path] = {}
    for split in SPLITS:
        raw_value = None
        for alias in SPLIT_ALIASES[split]:
            if raw.get(alias):
                raw_value = raw[alias]
                break
        images_dir = resolve_split_dir(base, raw_value, split)
        if images_dir is None and base != dataset_root:
            images_dir = resolve_split_dir(dataset_root, raw_value, split)
        if images_dir is None:
            continue
        try:
            has_image = any(
                p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                for p in images_dir.iterdir()
            )
        except OSError:
            has_image = False
        if has_image:
            found[split] = images_dir
    return found


def find_extracted_dataset(search_roots: Iterable[Path]) -> Path | None:
    """Zaten çıkarılmış, KULLANILABİLİR bir veri setinin data.yaml yolunu bulur.

    data.yaml'ın var olması yetmez; en az ``train`` bölümünün diskte görüntü
    içermesi aranır. Böylece yarım/boş klasörler "hazır" sanılmaz ve veri seti
    mevcutken tekrar indirme yapılmaz.
    """
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        data_yaml = find_data_yaml(root)
        if data_yaml is None:
            continue
        if "train" in dataset_splits_present(data_yaml):
            return data_yaml
    return None


# ==========================================================================
# Kaggle üzerinden indirme (yalnızca veri seti yoksa)
# ==========================================================================
def download_kaggle_dataset(
    handle: str = KAGGLE_DATASET_HANDLE,
    dest_dir: Path | None = None,
) -> Path:
    """kagglehub ile veri setini indirir ve data.yaml yolunu döndürür.

    ``dest_dir`` verilirse indirilen içerik oraya (ör. ``data/source``)
    kopyalanır; verilmezse kagglehub önbellek yolu kullanılır.
    """
    try:
        import kagglehub  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise RuntimeError(
            "kagglehub kurulu değil. Kurulum: pip install kagglehub | "
            "Kimlik: https://www.kaggle.com/settings -> Create New Token, "
            "inen kaggle.json dosyasını ~/.kaggle/ altına koyun."
        ) from exc

    cache_root = Path(kagglehub.dataset_download(handle))
    data_yaml = find_data_yaml(cache_root)
    if data_yaml is None:
        raise RuntimeError(
            f"İndirilen veri setinde data.yaml bulunamadı: {cache_root}"
        )

    if dest_dir is None:
        return data_yaml

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(data_yaml.parent, dest_dir, dirs_exist_ok=True)
    copied = find_data_yaml(dest_dir)
    return copied if copied is not None else data_yaml


# ==========================================================================
# Güvenli ZIP çıkarma
# ==========================================================================
def safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """ZIP'i zip-slip saldırılarına karşı güvenli biçimde çıkarır.

    Hedef klasörün dışına çıkmaya çalışan girdilerde hata verir.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (dest_resolved / member.filename).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise ValueError(
                    f"Güvenli olmayan ZIP girdisi engellendi: {member.filename}"
                )
        zf.extractall(dest_resolved)
    return dest_dir


# ==========================================================================
# data.yaml okuma / düzeltme
# ==========================================================================
def read_data_yaml(data_yaml: Path) -> dict:
    with Path(data_yaml).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_class_names(data_yaml: Path) -> list[str]:
    """``data.yaml`` içindeki sınıf isimlerini index sırasıyla döndürür.

    Hem liste (``names: [a, b]``) hem sözlük (``names: {0: a, 1: b}``)
    biçimini destekler.
    """
    data = read_data_yaml(data_yaml)
    names = data.get("names")
    if names is None:
        raise ValueError(f"'names' alanı bulunamadı: {data_yaml}")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise ValueError(f"'names' alanı beklenmeyen tipte ({type(names)}): {data_yaml}")


def resolve_split_dir(dataset_root: Path, raw_value: Any, split: str) -> Path | None:
    """data.yaml'daki split yolunu gerçek klasöre çözer.

    Mutlak/bozuk yollar, ``../`` önekleri ve eksik ``images`` alt klasörü
    tolere edilir; bulunamazsa klasör adı ile arama yapılır.
    """
    dataset_root = Path(dataset_root)
    candidates: list[Path] = []

    if raw_value:
        raw = Path(str(raw_value))
        # Mutlak yol: yalnızca gerçekten varsa kullan.
        if raw.is_absolute():
            candidates.append(raw)
            # Değilse son 1-2 parçasını proje köküne göre dene.
            candidates.append(dataset_root / raw.name)
            if len(raw.parts) >= 2:
                candidates.append(dataset_root / raw.parts[-2] / raw.parts[-1])
        else:
            parts = [p for p in raw.parts if p not in ("..", ".")]
            if parts:
                candidates.append(dataset_root.joinpath(*parts))
                candidates.append(dataset_root / parts[-1])
                if len(parts) >= 2:
                    candidates.append(dataset_root / parts[-2] / parts[-1])

    # Yaygın YOLO düzenleri
    for alias in SPLIT_ALIASES.get(split, (split,)):
        candidates.append(dataset_root / alias / "images")
        candidates.append(dataset_root / "images" / alias)
        candidates.append(dataset_root / alias)

    for cand in candidates:
        if cand.is_dir():
            # images alt klasörü varsa onu tercih et
            if (cand / "images").is_dir():
                return (cand / "images").resolve()
            return cand.resolve()
    return None


def labels_dir_for(images_dir: Path) -> Path:
    """Bir images klasörünün karşılık gelen labels klasörünü döndürür."""
    images_dir = Path(images_dir)
    if images_dir.name == "images":
        sibling = images_dir.parent / "labels"
        if sibling.is_dir():
            return sibling
        # images/train -> labels/train düzeni
        alt = images_dir.parent.parent / "labels" / images_dir.name
        if alt.is_dir():
            return alt
        return sibling
    if images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name
    return images_dir.parent / "labels"


# ==========================================================================
# Doğrulama
# ==========================================================================
@dataclass
class SplitReport:
    """Tek bir bölümün (train/val/test) doğrulama sonucu."""

    split: str
    images_dir: Path | None = None
    labels_dir: Path | None = None
    image_count: int = 0
    label_count: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    corrupt_labels: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    class_image_counts: Counter = field(default_factory=Counter)
    class_box_counts: Counter = field(default_factory=Counter)
    out_of_range_class_ids: list[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return self.images_dir is not None and self.image_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "images_dir": str(self.images_dir) if self.images_dir else None,
            "labels_dir": str(self.labels_dir) if self.labels_dir else None,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "missing_labels": self.missing_labels[:50],
            "missing_labels_total": len(self.missing_labels),
            "orphan_labels": self.orphan_labels[:50],
            "orphan_labels_total": len(self.orphan_labels),
            "empty_labels": self.empty_labels[:50],
            "empty_labels_total": len(self.empty_labels),
            "corrupt_labels": self.corrupt_labels[:50],
            "corrupt_labels_total": len(self.corrupt_labels),
            "corrupt_images": self.corrupt_images[:50],
            "corrupt_images_total": len(self.corrupt_images),
            "out_of_range_class_ids": self.out_of_range_class_ids[:50],
            "class_image_counts": dict(self.class_image_counts),
            "class_box_counts": dict(self.class_box_counts),
        }


@dataclass
class DatasetReport:
    """Veri setinin tamamı için doğrulama raporu."""

    dataset_root: Path
    data_yaml: Path
    class_names: list[str] = field(default_factory=list)
    splits: dict[str, SplitReport] = field(default_factory=dict)
    duplicates_across_splits: list[dict[str, Any]] = field(default_factory=list)
    duplicates_within_split: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(s.image_count for s in self.splits.values())

    def total_class_box_counts(self) -> Counter:
        total: Counter = Counter()
        for s in self.splits.values():
            total.update(s.class_box_counts)
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "data_yaml": str(self.data_yaml),
            "class_names": self.class_names,
            "class_count": len(self.class_names),
            "total_images": self.total_images,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "splits": {k: v.as_dict() for k, v in self.splits.items()},
            "duplicates_across_splits": self.duplicates_across_splits[:100],
            "duplicates_across_splits_total": len(self.duplicates_across_splits),
            "duplicates_within_split": self.duplicates_within_split[:100],
            "duplicates_within_split_total": len(self.duplicates_within_split),
            "total_class_box_counts": dict(self.total_class_box_counts()),
        }


def _parse_label_file(path: Path, class_count: int) -> tuple[list[int], str | None]:
    """YOLO label dosyasını okur.

    Döndürür: (sınıf id listesi, hata mesajı|None)
    """
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        return [], f"okunamadı: {exc}"

    class_ids: list[int] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            return class_ids, f"satır {lineno}: eksik alan ({len(parts)})"
        try:
            cid = int(float(parts[0]))
            coords = [float(v) for v in parts[1:5]]
        except ValueError:
            return class_ids, f"satır {lineno}: sayısal olmayan değer"
        if any(not -0.01 <= c <= 1.01 for c in coords):
            return class_ids, f"satır {lineno}: normalize edilmemiş koordinat"
        if class_count and not 0 <= cid < class_count:
            return class_ids, f"satır {lineno}: geçersiz sınıf id {cid}"
        class_ids.append(cid)
    return class_ids, None


def _image_is_readable(path: Path) -> bool:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return path.stat().st_size > 0
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - bozuk görüntü her türlü hatayı verebilir
        return False


def validate_split(
    split: str,
    images_dir: Path | None,
    class_names: list[str],
    check_images: bool = True,
) -> SplitReport:
    """Bir bölümü doğrular: eşleşme, boş/bozuk label, sınıf dağılımı."""
    report = SplitReport(split=split)
    if images_dir is None or not Path(images_dir).is_dir():
        return report

    images_dir = Path(images_dir)
    labels_dir = labels_dir_for(images_dir)
    report.images_dir = images_dir
    report.labels_dir = labels_dir if labels_dir.is_dir() else None

    images = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    report.image_count = len(images)

    label_files: dict[str, Path] = {}
    if labels_dir.is_dir():
        label_files = {
            p.stem: p for p in sorted(labels_dir.glob("*.txt")) if p.is_file()
        }
    report.label_count = len(label_files)

    class_count = len(class_names)
    seen_label_stems: set[str] = set()

    for img in images:
        if check_images and not _image_is_readable(img):
            report.corrupt_images.append(img.name)

        lbl = label_files.get(img.stem)
        if lbl is None:
            report.missing_labels.append(img.name)
            continue
        seen_label_stems.add(img.stem)

        class_ids, err = _parse_label_file(lbl, class_count)
        if err is not None:
            report.corrupt_labels.append(f"{lbl.name} ({err})")
            if "geçersiz sınıf id" in err:
                report.out_of_range_class_ids.append(f"{lbl.name} ({err})")
            continue
        if not class_ids:
            report.empty_labels.append(lbl.name)
            continue

        for cid in set(class_ids):
            name = class_names[cid] if 0 <= cid < class_count else f"id_{cid}"
            report.class_image_counts[name] += 1
        for cid in class_ids:
            name = class_names[cid] if 0 <= cid < class_count else f"id_{cid}"
            report.class_box_counts[name] += 1

    for stem, lbl in label_files.items():
        if stem not in seen_label_stems and stem not in {i.stem for i in images}:
            report.orphan_labels.append(lbl.name)

    return report


def detect_duplicates(
    splits: dict[str, SplitReport],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dosya hash'i ile kopya görüntüleri tespit eder.

    Döndürür: (bölümler arası sızıntılar, bölüm içi kopyalar)
    """
    hash_map: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split, report in splits.items():
        if report.images_dir is None:
            continue
        for img in sorted(report.images_dir.iterdir()):
            if not (img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES):
                continue
            try:
                digest = file_hash(img)
            except OSError:
                continue
            hash_map[digest].append((split, img))

    across: list[dict[str, Any]] = []
    within: list[dict[str, Any]] = []
    for digest, entries in hash_map.items():
        if len(entries) < 2:
            continue
        split_set = {s for s, _ in entries}
        record = {
            "hash": digest,
            "files": [f"{s}/{p.name}" for s, p in entries],
            "splits": sorted(split_set),
        }
        if len(split_set) > 1:
            across.append(record)
        else:
            within.append(record)
    return across, within


def validate_dataset(
    data_yaml: Path,
    check_images: bool = True,
    check_duplicates: bool = True,
) -> DatasetReport:
    """Veri setinin tamamını doğrular ve rapor döndürür."""
    data_yaml = Path(data_yaml).resolve()
    dataset_root = data_yaml.parent
    raw = read_data_yaml(data_yaml)

    try:
        class_names = read_class_names(data_yaml)
    except ValueError as exc:
        report = DatasetReport(dataset_root=dataset_root, data_yaml=data_yaml)
        report.errors.append(str(exc))
        return report

    report = DatasetReport(
        dataset_root=dataset_root, data_yaml=data_yaml, class_names=class_names
    )

    declared_nc = raw.get("nc")
    if declared_nc is not None and int(declared_nc) != len(class_names):
        report.warnings.append(
            f"data.yaml içindeki nc={declared_nc}, names uzunluğu={len(class_names)} "
            "ile uyuşmuyor."
        )

    # data.yaml 'path' anahtarı varsa göreli yolların tabanı olur.
    base = dataset_root
    if raw.get("path"):
        p = Path(str(raw["path"]))
        candidate = p if p.is_absolute() else (dataset_root / p)
        if candidate.is_dir():
            base = candidate

    for split in SPLITS:
        raw_value = None
        for alias in SPLIT_ALIASES[split]:
            if raw.get(alias):
                raw_value = raw[alias]
                break
        images_dir = resolve_split_dir(base, raw_value, split)
        if images_dir is None and base != dataset_root:
            images_dir = resolve_split_dir(dataset_root, raw_value, split)
        split_report = validate_split(
            split, images_dir, class_names, check_images=check_images
        )
        report.splits[split] = split_report

        if split in ("train", "val"):
            if not split_report.exists:
                report.errors.append(
                    f"'{split}' bölümü bulunamadı veya boş "
                    f"(data.yaml değeri: {raw_value!r})."
                )
        elif not split_report.exists:
            report.warnings.append("'test' bölümü bulunamadı; değerlendirme val ile yapılacak.")

        if split_report.missing_labels:
            report.warnings.append(
                f"'{split}': {len(split_report.missing_labels)} görüntünün "
                "label dosyası yok."
            )
        if split_report.orphan_labels:
            report.warnings.append(
                f"'{split}': {len(split_report.orphan_labels)} label dosyasının "
                "görüntüsü yok."
            )
        if split_report.empty_labels:
            report.warnings.append(
                f"'{split}': {len(split_report.empty_labels)} label dosyası boş "
                "(arka plan görüntüsü olarak değerlendirilir)."
            )
        if split_report.corrupt_labels:
            report.errors.append(
                f"'{split}': {len(split_report.corrupt_labels)} bozuk label dosyası."
            )
        if split_report.corrupt_images:
            report.errors.append(
                f"'{split}': {len(split_report.corrupt_images)} bozuk görüntü."
            )

    if check_duplicates:
        across, within = detect_duplicates(report.splits)
        report.duplicates_across_splits = across
        report.duplicates_within_split = within
        if across:
            report.warnings.append(
                f"{len(across)} görüntü birden fazla bölümde bulunuyor "
                "(veri sızıntısı riski)."
            )

    return report


# ==========================================================================
# Taşınabilir data.yaml üretimi
# ==========================================================================
def write_portable_data_yaml(
    report: DatasetReport,
    output_path: Path,
) -> Path:
    """Yolları düzeltilmiş, taşınabilir bir data.yaml yazar.

    Kaynak veri seti DEĞİŞTİRİLMEZ; yalnızca yeni dosya oluşturulur.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent.resolve()

    payload: dict[str, Any] = {"path": str(base).replace("\\", "/")}
    for split in SPLITS:
        split_report = report.splits.get(split)
        if split_report is None or split_report.images_dir is None:
            continue
        try:
            rel = split_report.images_dir.resolve().relative_to(base)
            value = str(rel).replace("\\", "/")
        except ValueError:
            value = str(split_report.images_dir.resolve()).replace("\\", "/")
        payload[split] = value

    payload["nc"] = len(report.class_names)
    payload["names"] = list(report.class_names)

    header = (
        "# OilBottleVisionDemo tarafindan otomatik olusturuldu.\n"
        "# Kaynak: " + str(report.data_yaml).replace("\\", "/") + "\n"
        "# Yollar tasinabilir olacak sekilde duzeltildi.\n"
    )
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return output_path


# ==========================================================================
# Raporlama
# ==========================================================================
def distribution_dataframe(report: DatasetReport):
    """Sınıf dağılımını pandas DataFrame olarak döndürür."""
    import pandas as pd  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for class_name in report.class_names:
        row: dict[str, Any] = {"class_name": class_name}
        total_images = 0
        total_boxes = 0
        for split in SPLITS:
            sr = report.splits.get(split)
            imgs = int(sr.class_image_counts.get(class_name, 0)) if sr else 0
            boxes = int(sr.class_box_counts.get(class_name, 0)) if sr else 0
            row[f"{split}_images"] = imgs
            row[f"{split}_boxes"] = boxes
            total_images += imgs
            total_boxes += boxes
        row["total_images"] = total_images
        row["total_boxes"] = total_boxes
        rows.append(row)
    return pd.DataFrame(rows)


def write_distribution_reports(report: DatasetReport, reports_dir: Path) -> dict[str, Path]:
    """Sınıf dağılımını CSV ve PNG grafik olarak kaydeder."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    df = distribution_dataframe(report)
    csv_path = reports_dir / "dataset_class_distribution.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    outputs["csv"] = csv_path

    # Bölüm özeti
    import pandas as pd  # noqa: PLC0415

    split_rows = []
    for split in SPLITS:
        sr = report.splits.get(split)
        if sr is None:
            continue
        split_rows.append(
            {
                "split": split,
                "images": sr.image_count,
                "labels": sr.label_count,
                "missing_labels": len(sr.missing_labels),
                "orphan_labels": len(sr.orphan_labels),
                "empty_labels": len(sr.empty_labels),
                "corrupt_labels": len(sr.corrupt_labels),
                "corrupt_images": len(sr.corrupt_images),
            }
        )
    split_csv = reports_dir / "dataset_split_summary.csv"
    pd.DataFrame(split_rows).to_csv(split_csv, index=False, encoding="utf-8-sig")
    outputs["split_csv"] = split_csv

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return outputs

    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bottom = None
        for split, color in (("train", "#2A6FB0"), ("val", "#F0A202"), ("test", "#7A9E7E")):
            col = f"{split}_boxes"
            if col not in df.columns:
                continue
            values = df[col].to_numpy()
            ax.bar(df["class_name"], values, bottom=bottom, label=split, color=color)
            bottom = values if bottom is None else bottom + values
        ax.set_title("BOTTLE_QA - sinif basina annotation (kutu) sayisi")
        ax.set_ylabel("Annotation sayisi")
        ax.set_xlabel("Sinif")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        chart_path = reports_dir / "dataset_class_distribution.png"
        fig.savefig(chart_path, dpi=140)
        plt.close(fig)
        outputs["chart"] = chart_path

    return outputs
