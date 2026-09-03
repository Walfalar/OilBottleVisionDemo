# OilBottleVisionDemo

Yağ fabrikası üretim hattındaki şişelerin **görüntü işleme ile kalite kontrolünü**
yapan, çalışan bir demo. Yüklenen fotoğraf veya videoyu inceleyip ürünü dört
karardan biriyle sınıflandırır:

| Karar | Renk | Anlamı |
|---|---|---|
| **SAĞLAM** | 🟢 yeşil | Şişe algılandı, kusur bulunamadı |
| **HATALI** | 🔴 kırmızı | En az bir kusur tespit edildi |
| **YANLIŞ ÜRÜN** | 🟠 turuncu | Farklı marka şişe |
| **MANUEL KONTROL GEREKLİ** | 🟡 sarı | Şişe algılanamadı veya güven değeri kararsız bantta |

> Bu bir **demodur**. Modelin doğruluğu kullanılan veri setiyle sınırlıdır ve
> hiçbir koşulda hatasız sonuç garanti etmez. Üretim hattında operatör
> doğrulaması olmadan kullanılmamalıdır.

---

## 1. Kullanılan veri seti

**Water Bottle Defect-Level Detection** (Kaggle) — 1.500 görüntü, YOLO formatı:

<https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset>

Kaggle handle: `jihadakbr/water-bottle-defect-level-detection-dataset`

> Not: Proje başlangıçta Roboflow Universe'teki BOTTLE_QA veri setine
> dayanıyordu; Roboflow indirme bağlantısı hata verdiği için aynı 7 sınıflı
> Kaggle sürümüne geçildi. Kod artık Roboflow'a bağımlı değildir.

Bölümler: `train` 1.200 görüntü / 1.200 label, `val` 300 / 300, ayrı `test`
bölümü **yok** (değerlendirme `val` ile yapılır).

Yerleşim (`data/source/` altında beklenir):

```
data/source/
├── data.yaml
├── images/{train,val}/
└── labels/{train,val}/
```

`scripts/prepare_dataset.py` bu klasörü **otomatik bulur**; veri seti
mevcutsa yeniden indirmez/çıkarmaz. Yoksa sırasıyla `data/source/*.zip`
dosyasını, o da yoksa `kagglehub` ile Kaggle indirmesini dener:

```powershell
pip install kagglehub          # tek seferlik
python scripts/prepare_dataset.py           # otomatik bul -> gerekirse indir
python scripts/prepare_dataset.py --download    # indirmeyi zorla
python scripts/prepare_dataset.py --no-download # sadece yereli kullan
```

Sınıflar (7 adet) ve karar eşlemesi:

| Sınıf (data.yaml) | Türkçe açıklama | Karar | Ciddiyet |
|---|---|---|---|
| `defect_debris` | Yabancı Madde | HATALI | 5 |
| `defect_no_cap` | Kapak Eksik | HATALI | 5 |
| `defect_loose_cap` | Gevşek/Hasarlı Kapak | HATALI | 4 |
| `defect_low_fill` | Eksik Dolum | HATALI | 4 |
| `defect_label_damage` | Etiket Hasarı | HATALI | 3 |
| `bottle_other_brand` | Yanlış Ürün / Farklı Marka | YANLIŞ ÜRÜN | 2 |
| `bottle_ron88` | Şişe (Ron88) | SAĞLAM | 0 |

**Sınıf indeksleri hiçbir yerde tahmin edilmez** — isimler her zaman veri
setinin `data.yaml` dosyasından okunur. Eşleme tablosu
`configs/project.yaml` → `class_mapping` altındadır. Tabloda karşılığı olmayan
bir sınıf gelirse güvenli tarafta kalınır ve **MANUEL KONTROL** kararı üretilir.

Bir görüntüde birden fazla kusur varsa **hepsi listelenir**; nihai karar
**en yüksek ciddiyetli kusura** göre HATALI olur.

---

## 2. Kurulum

### Gereksinimler

- **Python 3.11** (PyTorch/Ultralytics 3.12+ ve 3.14 üzerinde sorun çıkarabilir)
- NVIDIA GPU (opsiyonel — yoksa otomatik CPU'ya düşer)

### Windows (PowerShell)

```powershell
cd OilBottleVisionDemo

# Sanal ortam
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# PyTorch - GPU'lu kurulum (CUDA 12.8; RTX 40/50 serisi dahil)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# GPU YOKSA bunun yerine:
# pip install torch torchvision

# Kalan bağımlılıklar
pip install -r requirements.txt
```

> `Activate.ps1` çalıştırma politikası hatası verirse:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Ubuntu / Linux (bash)

```bash
cd OilBottleVisionDemo

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# PyTorch - GPU'lu kurulum
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# GPU YOKSA:
# pip install torch torchvision

pip install -r requirements.txt
```

### GPU kontrolü

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Daha ayrıntılı donanım raporu (veri seti kontrolüyle birlikte):

```bash
python scripts/inspect_dataset.py
```

### CPU fallback

Kod her yerde `device=auto` kullanır: GPU varsa `0`, yoksa `cpu`. Zorlamak için:

```bash
python scripts/train.py --device cpu
python scripts/predict_image.py --image foto.jpg --device cpu
```

`configs/project.yaml` → `model.device` ile kalıcı olarak da ayarlanabilir.
CPU'da eğitim çok yavaştır; CPU'da yalnızca çıkarım (inference) önerilir.

---

## 3. Veri setini yerleştirme

Veri setini yukarıdaki bağlantıdan **YOLOv8/YOLOv11 formatında ZIP** olarak
indirin ve tam olarak şu konuma koyun:

```
OilBottleVisionDemo/data/source/bottle_qa.zip
```

> Klasör yoksa oluşturun (`mkdir -p data/source` / `New-Item -ItemType Directory -Force data/source`)
> ya da bir kez `python scripts/inspect_dataset.py` çalıştırın — gerekli tüm
> klasörleri kendisi oluşturur.

Alternatif olarak `--zip` ile başka bir yol da verilebilir. Veri seti Git
deposuna dâhil **edilmez** (`.gitignore` içinde).

### Hazırlama

```bash
python scripts/prepare_dataset.py
# Mevcut kopyayı silip yeniden çıkarmak için:
python scripts/prepare_dataset.py --force
# Farklı bir ZIP:
python scripts/prepare_dataset.py --zip "C:/indirilenler/bottle_qa.zip"
```

Bu komut sırasıyla:

1. ZIP'i `data/prepared/bottle_qa` altına **güvenli biçimde** çıkarır (zip-slip koruması),
2. `data.yaml` dosyasını otomatik bulur ve sınıf isimlerini okur,
3. `train` / `val` / `test` klasörlerinin gerçekten var olduğunu doğrular,
4. görüntü–label eşleşmesini kontrol eder,
5. boş, bozuk veya karşılığı olmayan annotation dosyalarını raporlar,
6. sınıf başına görüntü ve annotation sayısını hesaplar,
7. dağılımı CSV ve PNG grafik olarak `outputs/reports` altına yazar,
8. dosya hash'leriyle **train/val/test arası kopya (veri sızıntısı)** arar,
9. `data.yaml` içindeki mutlak/bozuk yolları taşınabilir hâle getirip
   `data_portable.yaml` olarak yazar,
10. **kaynak veri setini değiştirmez.**

Salt okunur inceleme için:

```bash
python scripts/inspect_dataset.py
```

---

## 4. Eğitim

Varsayılan model **`yolo26n.pt`**; kurulu Ultralytics sürümü YOLO26'yı
desteklemiyorsa otomatik olarak **`yolo11n.pt`**'ye düşülür (görev durmaz).

```bash
# 3 epoch smoke test (her şeyin çalıştığını doğrular)
python scripts/train.py --smoke

# Ana eğitim (varsayılan 50 epoch, imgsz=640, patience=10, seed=42)
python scripts/train.py

# Parametre örnekleri
python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16 --imgsz 640 --device 0
```

| Bayrak | Anlamı |
|---|---|
| `--model` | Model ağırlığı (`yolo26n.pt`, `yolo11n.pt`, `yolo26s.pt`, ...) |
| `--epochs` | Epoch sayısı |
| `--smoke` | 3 epoch hızlı doğrulama |
| `--batch` | Batch size (verilmezse GPU belleğine göre otomatik) |
| `--imgsz` | Görüntü boyutu (varsayılan 640) |
| `--device` | `auto` / `cpu` / `0` |
| `--patience` | Early stopping sabrı |
| `--seed` | Tekrarlanabilirlik tohumu |

**Batch size:** GPU'da varsayılan `-1` (Ultralytics VRAM'e göre otomatik seçer).
`configs/project.yaml` → `training.batch` ile sabitlenebilir; CPU'da otomatik
olarak küçük bir değer kullanılır.

**Augmentation** — dolum seviyesi ve etiket kusurları renk/konum hassasiyeti
taşıdığı için bilinçli olarak ölçülü tutulmuştur
(`configs/project.yaml` → `training.augment`):

- dikey flip **kapalı** (`flipud: 0.0`)
- düşük oranda yatay flip (`fliplr: 0.2`)
- küçük rotasyon (`degrees: 5.0`), küçük öteleme/ölçek
- hafif parlaklık/kontrast/doygunluk (`hsv_v: 0.20`, `hsv_s: 0.30`)
- düşük mosaic (`mosaic: 0.1`, son 10 epoch kapalı)
- aşırı crop, shear, perspective, mixup, copy-paste **kapalı**

Eğitim çıktıları:

- `outputs/training/<run_adı>/` — grafikler, log'lar, ağırlıklar
- `models/best.pt` ve `models/last.pt` — kopyalanmış ağırlıklar
- `outputs/training/<run_adı>/training_report.json` — model, süre, cihaz,
  veri seti, en iyi epoch ve metrikler

---

## 5. Değerlendirme

```bash
python scripts/evaluate.py                         # test bölümü
python scripts/evaluate.py --split val
python scripts/evaluate.py --weights models/best.pt --no-error-analysis
```

Üretilenler (`outputs/reports/`):

| Dosya | İçerik |
|---|---|
| `evaluation_report.json` | Precision, Recall, F1, mAP50, mAP50-95 + sınıf bazlı tüm metrikler |
| `evaluation_per_class.csv` | Sınıf bazlı precision / recall / F1 / mAP |
| `confusion_matrix.png`, `confusion_matrix_normalized.png`, `confusion_matrix.csv` | Karışıklık matrisi |
| `false_positives.csv` | Yanlış pozitif örnekleri |
| `missed_defects.csv` | **Kaçırılmış kusur örnekleri** |
| `decision_audit.csv` | Görüntü bazlı gerçek/tahmin/karar dökümü |

> **Kalite kontrolde en kritik hata, kusurlu ürünün SAĞLAM kabul edilmesidir.**
> Bu yüzden rapor genel accuracy'e değil, **her kusur sınıfının recall değerine**
> odaklanır; `missed_defects.csv` ve konsoldaki "EN ZAYIF KUSUR" satırı
> öncelikli izlenmelidir.

---

## 6. Streamlit arayüzü

```bash
streamlit run app.py
```

Tarayıcıda <http://localhost:8501> açılır. Sekmeler:

1. **Ana Sayfa** — proje adı, model durumu, kullanılan ağırlık, çalışılan cihaz,
   confidence/IoU/manuel kontrol aralığı ayarları, donanım ve karar kuralları.
2. **Görüntü Analizi** — JPG/JPEG/PNG yükleme, orijinal ve bounding box'lı sonuç
   yan yana, karar rozeti, bulunan kusurlar, confidence değerleri, işlem süresi,
   sonucu indirme.
3. **Video Analizi** — MP4 yükleme, frame bazlı analiz, analiz edilmiş videoyu
   kaydetme, toplam incelenen ürün / sağlam / hatalı sayıları, kusur türlerinin
   dağılım grafiği, ürün bazlı CSV indirme.
4. **Geçmiş Kayıtlar** — tarih-saat, dosya adı, nihai karar, kusur türleri, en
   yüksek confidence, işlem süresi; filtreleme ve CSV dışa aktarma.

Hatalı bulunan görüntüler bounding box çizilmiş hâlde **`outputs/rejected/`**
klasörüne kaydedilir.

---

## 7. Tahmin (komut satırı)

### Görüntü

```bash
python scripts/predict_image.py --image data/prepared/bottle_qa/test/images/ornek.jpg
python scripts/predict_image.py --image klasor/ --json sonuclar.json
python scripts/predict_image.py --image ornek.jpg --conf 0.35 --device cpu
```

Sonucu terminalde Türkçe yazar, annotated görüntüyü kaydeder (`HATALI` ise
`outputs/rejected/`, aksi hâlde `outputs/predictions/`) ve `--json` ile JSON
çıktı üretir.

### Video

```bash
python scripts/predict_video.py --video hat_kaydi.mp4
python scripts/predict_video.py --video hat_kaydi.mp4 --stride 2 --max-frames 500
python scripts/predict_video.py --video hat_kaydi.mp4 --no-video-out
```

Aynı şişenin ardışık frame'lerde tekrar tekrar sayılmaması için **IoU tabanlı
basit takip + zaman tabanlı tekrar engelleme** kullanılır
(`configs/project.yaml` → `video`). Bir ürün için birden fazla frame'de farklı
kararlar çıkarsa **en kötü karar** geçerli olur (kusuru kaçırmamak için).

Çıktılar: annotated MP4, `*_ozet.json`, `*_urunler.csv`, `*_frameler.csv`
(`outputs/predictions/`) ve hatalı ürün kareleri (`outputs/rejected/`).

---

## 8. Testler

```bash
pytest tests -q
pytest tests -v
```

Kapsanan senaryolar:

- Bir kusur bulunduğunda karar HATALI oluyor mu (beş kusur sınıfı ayrı ayrı)
- Birden fazla kusur doğru listeleniyor mu, en ciddi kusur karara adını veriyor mu
- `bottle_other_brand` → YANLIŞ ÜRÜN
- Hiç nesne yoksa → MANUEL KONTROL
- Sağlam şişe doğru değerlendiriliyor mu
- Kararsız güven bandı → MANUEL KONTROL (güvenilir şişe + kararsız kusur asla SAĞLAM olmaz)
- Bilinmeyen sınıf asla SAĞLAM sayılmıyor
- `data.yaml` sınıfları okunuyor mu (liste ve sözlük biçimi)
- Eksik görüntü/label eşleşmesi, boş/bozuk label, geçersiz sınıf id tespiti
- Bölümler arası kopya (veri sızıntısı) tespiti
- Bozuk yolların taşınabilir `data.yaml`'a düzeltilmesi, kaynağın değiştirilmemesi
- ZIP güvenli çıkarma ve zip-slip engelleme

---

## 9. Çıktı klasörleri

```
data/source/        indirilen ZIP dosyası (Git'e girmez)
data/prepared/      çıkarılmış + doğrulanmış veri seti (Git'e girmez)
models/             best.pt, last.pt
outputs/training/   eğitim çalışmaları, grafikler, training_report.json
outputs/predictions/annotated görüntü ve videolar, özet JSON/CSV
outputs/rejected/   HATALI bulunan ürünlerin bounding box'lı görüntüleri
outputs/reports/    veri seti ve değerlendirme raporları, grafikler
logs/inspections.csv her kontrolün kaydı
```

`logs/inspections.csv` sütunları: `timestamp`, `source_type`, `source_name`,
`final_status`, `defect_classes`, `confidence`, `inference_time_ms`,
`output_path`.

---

## 10. Yapılandırma

Tüm eşikler ve parametreler `configs/project.yaml` üzerinden ayarlanır:

```yaml
thresholds:
  detection_confidence: 0.25   # bu değerin üstü "güvenilir" tespit
  manual_review_low:    0.15   # bu değerin altı gürültü sayılır
  manual_review_high:   0.30   # low–high arası MANUEL KONTROL üretir
  iou_threshold:        0.45
```

Karar mantığı:

| Durum | Karar |
|---|---|
| Güvenilir kusur var | **HATALI** (en ciddi kusura göre) |
| Güvenilir `bottle_other_brand` | **YANLIŞ ÜRÜN** |
| Sadece kararsız bantta tespit var | **MANUEL KONTROL** |
| Hiç anlamlı tespit yok | **MANUEL KONTROL** |
| Şişe var, kusur yok | **SAĞLAM** |

Farklı bir yapılandırma dosyası kullanmak için `OILBOTTLE_CONFIG` ortam
değişkeni ayarlanabilir.

---

## 11. Mevcut durum ve doğrulama

Bu depo teslim edildiğinde **BOTTLE_QA veri seti workspace'te bulunamamıştır**,
bu nedenle `models/` klasörü **boştur** ve gerçek veriyle eğitilmiş bir model
yoktur. Veri setini yukarıdaki adımlarla yerleştirip
`prepare_dataset.py` → `train.py` çalıştırdığınızda `models/best.pt` oluşur.

Kod yolu, gerçek veri yerine kontrollü bir doğrulama veri setiyle uçtan uca
çalıştırılarak doğrulanmıştır: veri hazırlama, 3 epoch smoke test, 40 epoch
eğitim (RTX 5060, CUDA 12.8, YOLO26n), değerlendirme raporları, tek görüntü ve
video çıkarımı, Streamlit arayüzü ve 56 birim testi. Doğrulama sonrası üretilen
tüm geçici çıktılar ve ağırlıklar, gerçek veriden gelmedikleri için silinmiştir.

**Bu depodaki hiçbir metrik gerçek BOTTLE_QA performansını temsil etmez.**
Gerçek metrikler eğitimden sonra `outputs/reports/evaluation_report.json`
dosyasında oluşacaktır.

---

## 12. Bilinen sınırlamalar

- Demo, **BOTTLE_QA** açık veri setiyle hazırlanmıştır; fabrikanın kendi
  aydınlatma, kamera açısı, konveyör hızı ve şişe tipiyle **eşleşmeyebilir**.
- Veri seti ~1.500 görüntüdür ve bazı kusur sınıflarında örnek sayısı azdır;
  bu sınıflarda recall düşük kalabilir.
- Nano model (`yolo26n` / `yolo11n`) hız için seçilmiştir; doğruluk `s`/`m`
  modellerine göre daha düşüktür.
- Video takibi **basit bir IoU + zaman tabanlı** yöntemdir. Hızlı hat, çoklu
  şerit veya birbirini örten şişelerde sayım hatalı olabilir; gerçek hat için
  ByteTrack/BoT-SORT gibi bir takipçi gerekir.
- Analiz videosu `mp4v` codec ile yazılır; bazı tarayıcılar bu codec'i
  oynatamayabilir (dosya yine de diske doğru yazılır).
- Yüzey ezilmesi, şişe deformasyonu ve yağ içindeki çok küçük yabancı maddeler
  **bu fazın kapsamı dışındadır** (2. faz).
- Sistem hiçbir koşulda hatasız sonuç **garanti etmez**; MANUEL KONTROL kararı
  operatöre yönlendirme amaçlıdır.

---

## 13. Fabrika verisine geçişte yapılacaklar

1. **Veri toplama:** Gerçek hattan, gerçek aydınlatma ve kamera konumuyla her
   kusur sınıfı için en az birkaç yüz örnek toplayın; özellikle nadir kusurları
   bilinçli olarak çoğaltın.
2. **Etiketleme:** Aynı sınıf isimlerini koruyun (`defect_*`, `bottle_*`).
   İsimler değişirse `configs/project.yaml` → `class_mapping` güncellenmelidir;
   kod sınıfları `data.yaml`'dan okuduğu için başka değişiklik gerekmez.
3. **Bölme:** Train/val/test ayrımını **ürün partisi/vardiya bazında** yapın;
   `prepare_dataset.py` hash tabanlı sızıntı kontrolünü otomatik çalıştırır.
4. **Yeniden eğitim:** `python scripts/prepare_dataset.py` → `--smoke` →
   ana eğitim. Daha yüksek doğruluk için `--model yolo26s.pt` veya `yolo11s.pt`.
5. **Eşik kalibrasyonu:** `evaluate.py` çıktısındaki `missed_defects.csv`'ye
   bakarak `detection_confidence` ve manuel kontrol aralığını ayarlayın.
   Kusur kaçırma maliyeti yüksekse eşiği **düşürüp** manuel kontrol payını
   artırın.
6. **Takip:** Hat hızı yüksekse `predict_video.py` içindeki basit takibi gerçek
   bir çoklu nesne takipçisiyle değiştirin.
7. **Entegrasyon:** `logs/inspections.csv` yerine bir veritabanı/PLC bağlantısı;
   `src/decision_engine.py` kararı doğrudan ayıklama (reject) mekanizmasına
   bağlanabilir.
8. **İzleme:** Üretimde kusur oranı ve MANUEL KONTROL oranını takip edin; ani
   artış model kaymasının (drift) ilk işaretidir.

---

## 14. Lisans uyarıları

- **Veri seti:** *Water Bottle Defect-Level Detection*, Kaggle üzerinde
  (`jihadakbr/water-bottle-defect-level-detection-dataset`) yayımlanmıştır.
  Kullanım koşulları ve lisansı veri setinin kendi sayfasından doğrulanmalıdır;
  ticari kullanım öncesi bu şart zorunludur. Veri seti bu depoya dâhil edilmez.
- **Ultralytics YOLO:** **AGPL-3.0** ile dağıtılır. AGPL, türev çalışmaların da
  aynı lisansla yayımlanmasını gerektirir. Kapalı kaynak/ticari bir üründe
  kullanmadan önce Ultralytics'ten **ticari lisans** alınması gerekir.
- **PyTorch (BSD-3), OpenCV (Apache-2.0), Streamlit (Apache-2.0), pandas
  (BSD-3), NumPy (BSD-3), Pillow (MIT-CMU)** — izin verici lisanslar.
- Bu depodaki kod fabrikaya özel iş mantığı içerir; dağıtımı kurum politikasına
  tabidir.
