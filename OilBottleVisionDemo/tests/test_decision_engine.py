"""Karar motoru testleri.

Kalite kontrolde en kritik hata kusurlu ürünün SAĞLAM sayılmasıdır; testler
özellikle bu senaryoyu kapsar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.decision_engine import (  # noqa: E402
    STATUS_DEFECT,
    STATUS_MANUAL,
    STATUS_OK,
    STATUS_WRONG_PRODUCT,
    DecisionEngine,
    Detection,
)

BOX = (10.0, 10.0, 100.0, 200.0)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def engine(config):
    return DecisionEngine(config=config)


def det(name: str, conf: float = 0.9, class_id: int = 0) -> Detection:
    return Detection(class_id=class_id, class_name=name, confidence=conf, bbox=BOX)


# --------------------------------------------------------------------------
# Sağlam ürün
# --------------------------------------------------------------------------
def test_saglam_sise_dogru_degerlendiriliyor(engine):
    """Şişe algılandı, kusur yok -> SAĞLAM."""
    result = engine.evaluate([det("bottle_ron88", 0.92)])
    assert result.status_key == STATUS_OK
    assert result.status == "SAĞLAM"
    assert result.defects == []
    assert result.is_defective is False
    assert result.color_hex == "#1B9E4B"  # yeşil


# --------------------------------------------------------------------------
# Tek kusur -> HATALI
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("class_name", "expected_label"),
    [
        ("defect_debris", "Yabancı Madde"),
        ("defect_label_damage", "Etiket Hasarı"),
        ("defect_loose_cap", "Gevşek/Hasarlı Kapak"),
        ("defect_low_fill", "Eksik Dolum"),
        ("defect_no_cap", "Kapak Eksik"),
    ],
)
def test_tek_kusur_hatali_karari_uretir(engine, class_name, expected_label):
    """Herhangi bir kusur sınıfı algılandığında karar HATALI olmalı."""
    result = engine.evaluate([det("bottle_ron88", 0.9), det(class_name, 0.8)])
    assert result.status_key == STATUS_DEFECT
    assert result.status == "HATALI"
    assert result.is_defective is True
    assert expected_label in result.defect_labels
    assert class_name in result.defect_classes
    assert result.color_hex == "#D62828"  # kırmızı


def test_kusur_tek_basina_da_hatali(engine):
    """Şişe sınıfı olmadan yalnız kusur algılansa bile karar HATALI."""
    result = engine.evaluate([det("defect_no_cap", 0.77)])
    assert result.status_key == STATUS_DEFECT


# --------------------------------------------------------------------------
# Birden fazla kusur
# --------------------------------------------------------------------------
def test_birden_fazla_kusur_hepsi_listelenir(engine):
    """Bir görüntüdeki tüm kusurlar listelenmeli, karar HATALI olmalı."""
    result = engine.evaluate(
        [
            det("bottle_ron88", 0.95),
            det("defect_low_fill", 0.71),
            det("defect_label_damage", 0.64),
            det("defect_debris", 0.58),
        ]
    )
    assert result.status_key == STATUS_DEFECT
    assert set(result.defect_classes) == {
        "defect_low_fill",
        "defect_label_damage",
        "defect_debris",
    }
    assert len(result.defect_labels) == 3


def test_en_ciddi_kusur_karara_adini_verir(engine):
    """Nihai gerekçe en yüksek severity değerli kusurdan gelmeli."""
    result = engine.evaluate(
        [det("defect_label_damage", 0.95), det("defect_debris", 0.40)]
    )
    assert result.status_key == STATUS_DEFECT
    # defect_debris severity=5 > defect_label_damage severity=3
    assert result.reason == "Yabancı Madde"
    assert result.defect_labels[0] == "Yabancı Madde"


def test_ayni_kusur_tekrar_etse_bir_kez_listelenir(engine):
    result = engine.evaluate([det("defect_debris", 0.8), det("defect_debris", 0.6)])
    assert result.defect_classes == ["defect_debris"]


# --------------------------------------------------------------------------
# Yanlış ürün
# --------------------------------------------------------------------------
def test_other_brand_yanlis_urun_karari_uretir(engine):
    """bottle_other_brand -> YANLIŞ ÜRÜN."""
    result = engine.evaluate([det("bottle_other_brand", 0.88)])
    assert result.status_key == STATUS_WRONG_PRODUCT
    assert result.status == "YANLIŞ ÜRÜN"
    assert result.is_defective is False
    assert result.color_hex == "#F77F00"  # turuncu


def test_yanlis_urunde_kusur_varsa_hatali_oncelikli(engine):
    """Yanlış markalı şişede kusur da varsa karar HATALI olmalı."""
    result = engine.evaluate(
        [det("bottle_other_brand", 0.9), det("defect_no_cap", 0.85)]
    )
    assert result.status_key == STATUS_DEFECT


# --------------------------------------------------------------------------
# Manuel kontrol
# --------------------------------------------------------------------------
def test_hic_nesne_yoksa_manuel_kontrol(engine):
    """Hiç tespit yoksa -> MANUEL KONTROL GEREKLİ."""
    result = engine.evaluate([])
    assert result.status_key == STATUS_MANUAL
    assert result.status == "MANUEL KONTROL GEREKLİ"
    assert result.reason == "Şişe algılanamadı"
    assert result.max_confidence == 0.0
    assert result.color_hex == "#F2C500"  # sarı


def test_cok_dusuk_guven_yok_sayilir(engine):
    """manual_review_low altındaki tespitler gürültü sayılır -> MANUEL."""
    low = engine.thresholds.manual_review_low
    result = engine.evaluate([det("bottle_ron88", low - 0.05)])
    assert result.status_key == STATUS_MANUAL
    assert result.detections == []


def test_kararsiz_bant_manuel_kontrol_uretir(engine):
    """Güven değeri manuel inceleme aralığındaysa -> MANUEL KONTROL."""
    t = engine.thresholds
    mid = (t.manual_review_low + t.manual_review_high) / 2
    result = engine.evaluate([det("bottle_ron88", mid)])
    assert result.status_key == STATUS_MANUAL
    assert "Kararsız güven" in result.reason


def test_kararsiz_kusur_saglam_sayilmaz(engine):
    """Güvenilir şişe + kararsız kusur -> asla SAĞLAM olmamalı (kritik)."""
    t = engine.thresholds
    mid = (t.manual_review_low + t.manual_review_high) / 2
    result = engine.evaluate([det("bottle_ron88", 0.95), det("defect_debris", mid)])
    assert result.status_key == STATUS_MANUAL
    assert result.status_key != STATUS_OK


def test_guvenilir_kusur_kararsiz_tespite_ustun(engine):
    """Güvenilir kusur varsa kararsız tespitler kararı değiştirmez."""
    t = engine.thresholds
    mid = (t.manual_review_low + t.manual_review_high) / 2
    result = engine.evaluate([det("defect_debris", 0.9), det("bottle_ron88", mid)])
    assert result.status_key == STATUS_DEFECT


def test_bilinmeyen_sinif_saglam_sayilmaz(engine):
    """Eşleme tablosunda olmayan sınıf güvenli tarafta kalmalı."""
    result = engine.evaluate([det("yeni_kusur_tipi", 0.95)])
    assert result.status_key == STATUS_MANUAL
    assert result.status_key != STATUS_OK


# --------------------------------------------------------------------------
# Eşikler ve yapılandırma
# --------------------------------------------------------------------------
def test_varsayilan_esikler(config):
    t = config.thresholds
    assert t.detection_confidence == pytest.approx(0.25)
    assert t.manual_review_low == pytest.approx(0.15)
    assert t.manual_review_high == pytest.approx(0.30)
    assert t.iou_threshold == pytest.approx(0.45)
    assert t.validate() == []


def test_esikler_yapilandirilabilir(config):
    """Eşikler değiştirildiğinde karar da değişmeli."""
    from src.config import Thresholds  # noqa: PLC0415

    strict = Thresholds(
        detection_confidence=0.80, manual_review_low=0.70, manual_review_high=0.85
    )
    engine = DecisionEngine(config=config, thresholds=strict)
    # 0.75 -> sıkı eşiklerde kararsız bantta kalır
    assert engine.evaluate([det("bottle_ron88", 0.75)]).status_key == STATUS_MANUAL
    assert engine.evaluate([det("bottle_ron88", 0.90)]).status_key == STATUS_OK


def test_tutarsiz_esikler_uyari_verir():
    from src.config import Thresholds  # noqa: PLC0415

    bad = Thresholds(manual_review_low=0.9, manual_review_high=0.2)
    assert bad.validate()


def test_yapilandirmadaki_tum_veri_seti_siniflari_eslesmis(config):
    """BOTTLE_QA sınıflarının tamamı class_mapping içinde tanımlı olmalı."""
    beklenen = {
        "bottle_other_brand",
        "bottle_ron88",
        "defect_debris",
        "defect_label_damage",
        "defect_loose_cap",
        "defect_low_fill",
        "defect_no_cap",
    }
    engine = DecisionEngine(config=config)
    assert beklenen <= set(engine.rules)
    assert engine.unmapped_classes(sorted(beklenen)) == []


# --------------------------------------------------------------------------
# Serileştirme
# --------------------------------------------------------------------------
def test_sonuc_sozluge_cevrilebilir(engine):
    result = engine.evaluate(
        [det("bottle_ron88", 0.9), det("defect_low_fill", 0.7)], inference_time_ms=12.3
    )
    data = result.as_dict()
    assert data["status"] == "HATALI"
    assert data["is_defective"] is True
    assert "defect_low_fill" in data["defect_classes"]
    assert data["inference_time_ms"] == pytest.approx(12.3)
    assert len(data["detections"]) == 2


def test_evaluate_raw_demetlerden_calisir(engine):
    result = engine.evaluate_raw([(4, "defect_no_cap", 0.81, BOX)])
    assert result.status_key == STATUS_DEFECT


# --------------------------------------------------------------------------
# Video takibi - aynı şişenin tekrar tekrar sayılmaması
# --------------------------------------------------------------------------
from scripts.predict_video import ProductTracker  # noqa: E402


def _tracker() -> ProductTracker:
    return ProductTracker(
        iou_threshold=0.35,
        suppression_sec=1.5,
        center_distance_ratio=0.15,
        frame_size=(640, 640),
    )


def _kayan_kutu(x: float) -> tuple[float, float, float, float]:
    """Konveyörde sağa doğru kayan bir şişe kutusu."""
    return (x, 120.0, x + 180.0, 500.0)


def test_ayni_sise_ardisik_framelerde_bir_kez_sayilir(engine):
    """Kutular örtüşerek ilerlerken tek ürün sayılmalı."""
    tracker = _tracker()
    for i in range(8):
        result = engine.evaluate(
            [Detection(1, "bottle_ron88", 0.9, _kayan_kutu(40.0 + i * 40))]
        )
        tracker.update(result, time_s=i * 0.12, frame_index=i * 3)
    assert len(tracker.products) == 1
    assert tracker.products[0].frame_count == 8


def test_hizli_hatta_ortusmeyen_kutular_da_tek_urun(engine):
    """Kutu kendi boyutundan fazla ilerlese bile (IoU=0) aynı ürün sayılmalı.

    Küçük kusur kutusu hızlı hatta ardışık framelerde hiç örtüşmez; merkez
    mesafesi ölçütü olmadan aynı şişe defalarca sayılırdı.
    """
    tracker = _tracker()
    for i in range(6):
        kutu = (50.0 + i * 60, 400.0, 92.0 + i * 60, 442.0)  # 42 px kutu, 60 px adım
        result = engine.evaluate([Detection(2, "defect_debris", 0.7, kutu)])
        tracker.update(result, time_s=i * 0.12, frame_index=i * 3)
    assert len(tracker.products) == 1


def test_uzak_konumdaki_yeni_sise_ayri_urun(engine):
    """Hattın diğer ucunda beliren şişe yeni ürün olarak sayılmalı."""
    tracker = _tracker()
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.9, _kayan_kutu(420.0))]),
        time_s=0.0,
        frame_index=0,
    )
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.9, _kayan_kutu(10.0))]),
        time_s=0.24,
        frame_index=6,
    )
    assert len(tracker.products) == 2


def test_zaman_asimindan_sonra_yeni_urun(engine):
    """Tekrar engelleme süresi aşıldıysa aynı konumda bile yeni üründür."""
    tracker = _tracker()
    kutu = _kayan_kutu(200.0)
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.9, kutu)]), 0.0, 0
    )
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.9, kutu)]), 5.0, 125
    )
    assert len(tracker.products) == 2


def test_urun_icin_en_kotu_karar_gecerli(engine):
    """Bir framede kusur görülürse ürünün kararı HATALI kalmalı."""
    tracker = _tracker()
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.95, _kayan_kutu(40.0))]), 0.0, 0
    )
    tracker.update(
        engine.evaluate(
            [
                Detection(1, "bottle_ron88", 0.95, _kayan_kutu(80.0)),
                Detection(2, "defect_debris", 0.80, (150.0, 400.0, 190.0, 440.0)),
            ]
        ),
        0.12,
        3,
    )
    tracker.update(
        engine.evaluate([Detection(1, "bottle_ron88", 0.95, _kayan_kutu(120.0))]), 0.24, 6
    )

    assert len(tracker.products) == 1
    urun = tracker.products[0]
    assert urun.status_key == STATUS_DEFECT
    assert "defect_debris" in urun.defect_classes
    assert tracker.summary()["defective"] == 1
    assert tracker.summary()["total_products"] == 1


def test_tespit_yoksa_urun_sayilmaz(engine):
    tracker = _tracker()
    for i in range(5):
        tracker.update(engine.evaluate([]), time_s=i * 0.12, frame_index=i * 3)
    assert tracker.products == []
    assert tracker.summary()["total_products"] == 0
