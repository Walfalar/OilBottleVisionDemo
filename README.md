# TurnaAI

Turna AI — QualityVision.

## Kurulum

Nerede kalmıştık
Depo iskeleti (1 Eyl, commit'lendi)

git init -b main, ilk commit 6143956 — sadece kök .gitignore + README.md
Gitea'yı senin isteğinle boş bıraktık, uzak sunucu tanımlı değil
OilBottleVisionDemo (yazıldı, ama commit'lenmedi)

Yağ şişesi kalite kontrol demosu tamamen kodlanmış — 27 dosya, hepsi hâlâ untracked (?? OilBottleVisionDemo/).

Katman	Durum
src/	config, dataset, decision_engine, detector, logger, utils, visualization
scripts/	prepare_dataset, inspect_dataset, train, evaluate, predict_image, predict_video
app.py	Streamlit arayüzü (4 sekme)
configs/project.yaml	eşikler, augmentation, video takibi, 7 sınıflık karar eşlemesi
tests/	55 geçti, 1 atlandı (az önce çalıştırdım, 3 sn)
Dokümantasyon	14 bölümlük detaylı README
Ortam hazır ve GPU çalışıyor

.venv → Python 3.11.9, torch 2.11.0+cu128, ultralytics 8.4.137
CUDA True → RTX 5060 Laptop GPU
Ağırlıklar indirilmiş: yolo26n.pt, yolo11n.pt (+ weights/yolo26n.pt)
Eksik olan tek şey: veri seti
Boş klasörler: data/source/, data/prepared/, models/, outputs/training/. logs/inspections.csv sadece başlık satırından ibaret — yani hiç eğitim ve hiç çıkarım yapılmamış.

README'nin 11. bölümü de bunu söylüyor: BOTTLE_QA veri seti workspace'te bulunamadığı için model eğitilmemiş, doğrulama sentetik veriyle yapılıp çıktılar silinmiş.

Sıradaki adım
BOTTLE_QA ZIP'ini Roboflow'dan indirip OilBottleVisionDemo/data/source/bottle_qa.zip konumuna koymak, sonra:

Bir de şu var: demo klasörünün tamamı commit'lenmemiş durumda. İstersen önce onu commit'leyeyim mi?
