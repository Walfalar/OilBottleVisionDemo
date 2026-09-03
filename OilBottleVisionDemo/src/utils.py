"""Ortak yardımcı fonksiyonlar: donanım tespiti, hashing, konsol, zaman."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------------------
# Konsol / kodlama
# --------------------------------------------------------------------------
def configure_console() -> None:
    """Windows konsolunda Türkçe karakterlerin bozulmamasını sağlar."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, io.UnsupportedOperation):
            # Yeniden yapılandırılamayan akışlarda sessizce devam et.
            pass


# --------------------------------------------------------------------------
# Zaman
# --------------------------------------------------------------------------
def timestamp(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)


def file_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} sa {m} dk {s} sn"
    if m:
        return f"{m} dk {s} sn"
    return f"{seconds:.1f} sn"


@contextmanager
def stopwatch() -> Iterator[dict]:
    """`with stopwatch() as t:` -> t['ms'] işlem süresini milisaniye verir."""
    result: dict[str, float] = {"ms": 0.0, "s": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["s"] = elapsed
        result["ms"] = elapsed * 1000.0


# --------------------------------------------------------------------------
# Dosya işlemleri
# --------------------------------------------------------------------------
def file_hash(path: Path, algo: str = "md5", chunk_size: int = 1 << 20) -> str:
    """Dosyanın içerik hash'ini döndürür (kopya tespiti için)."""
    h = hashlib.new(algo)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(directory: Path) -> Iterator[Path]:
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def safe_mkdir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return path


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """Dosya adını Windows/Linux'ta güvenli hâle getirir."""
    invalid = r'<>:"/\|?*'
    cleaned = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in name)
    cleaned = cleaned.strip().strip(".")
    return (cleaned or "dosya")[:max_length]


# --------------------------------------------------------------------------
# Donanım
# --------------------------------------------------------------------------
def hardware_info() -> dict:
    """PyTorch varsa GPU bilgisini, yoksa yalnızca CPU bilgisini döndürür."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
    }
    try:
        import torch  # noqa: PLC0415  (isteğe bağlı bağımlılık)
    except ImportError:
        return info

    info["torch"] = torch.__version__
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError):
        info["cuda_available"] = False
    if info["cuda_available"]:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def resolve_device(requested: str | int | None = "auto") -> str:
    """İstenen cihazı çözer. 'auto' -> GPU varsa '0', yoksa 'cpu'."""
    if requested is None:
        requested = "auto"
    requested = str(requested).strip().lower()
    if requested and requested != "auto":
        return requested
    info = hardware_info()
    return "0" if info["cuda_available"] and info["gpus"] else "cpu"


def torch_device_string(device: str) -> str:
    """Ultralytics cihaz gösterimini PyTorch gösterimine çevirir.

    Ultralytics '0' / '0,1' kabul eder; ``model.to()`` ise 'cuda:0' bekler.
    """
    device = str(device).strip().lower()
    if device in ("cpu", "mps"):
        return device
    if device.startswith("cuda"):
        return device
    first = device.split(",")[0].strip()
    if first.isdigit():
        return f"cuda:{first}"
    return device or "cpu"


def gpu_total_memory_gb(device: str) -> float | None:
    """Seçili GPU'nun toplam belleğini GB olarak döndürür."""
    if device == "cpu":
        return None
    info = hardware_info()
    if not info["gpus"]:
        return None
    try:
        idx = int(str(device).split(",")[0])
    except ValueError:
        idx = 0
    for gpu in info["gpus"]:
        if gpu["index"] == idx:
            return float(gpu["total_memory_gb"])
    return float(info["gpus"][0]["total_memory_gb"])


def suggest_batch_size(device: str, imgsz: int = 640) -> int:
    """GPU belleğine göre kaba bir batch size önerir (yolo nano, imgsz=640)."""
    vram = gpu_total_memory_gb(device)
    if vram is None:
        return 4  # CPU
    scale = (640 / max(imgsz, 1)) ** 2
    if vram >= 22:
        base = 32
    elif vram >= 14:
        base = 24
    elif vram >= 10:
        base = 16
    elif vram >= 7:
        base = 12
    elif vram >= 5:
        base = 8
    else:
        base = 4
    return max(2, int(base * scale))
