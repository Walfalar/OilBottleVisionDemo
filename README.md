# TurnaAI

Turna AI — QualityVision. Bilgisayarlı görü tabanlı kalite kontrol çalışmaları.

## Projeler

| Klasör | Açıklama |
| --- | --- |
| [`OilBottleVisionDemo/`](OilBottleVisionDemo/) | YOLO tabanlı yağ şişesi kalite kontrol demosu — veri hazırlama, eğitim, değerlendirme, görüntü/video çıkarımı ve Streamlit arayüzü. |

## Hızlı başlangıç

Her proje kendi README'sinde ayrıntılı kurulum adımlarını içerir. `OilBottleVisionDemo` için:

```powershell
cd OilBottleVisionDemo

# Sanal ortam (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch — GPU'lu kurulum (CUDA 12.8; RTX 40/50 serisi dahil)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# GPU yoksa: pip install torch torchvision

pip install -r requirements.txt
```

Ayrıntılar: [`OilBottleVisionDemo/README.md`](OilBottleVisionDemo/README.md)

## Depoya dahil olmayanlar

`.gitignore` gereği aşağıdakiler depoda tutulmaz, her makinede yerel olarak edinilmelidir:

- Sanal ortamlar (`.venv/`)
- Model ağırlıkları (`*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.ckpt`)
- Veri setleri (`data/`, `datasets/`) ve eğitim çıktıları (`runs/`)
- Ortam dosyaları (`.env`)
