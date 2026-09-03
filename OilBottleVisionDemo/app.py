"""OilBottleVisionDemo - Streamlit kalite kontrol arayüzü.

Başlatma:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import Thresholds, load_config  # noqa: E402
from src.decision_engine import (  # noqa: E402
    STATUS_COLORS_HEX,
    STATUS_DEFECT,
    STATUS_LABELS,
    STATUS_MANUAL,
    STATUS_OK,
    STATUS_WRONG_PRODUCT,
    InspectionResult,
)
from src.detector import BottleDetector, ModelLoadError  # noqa: E402
from src.logger import InspectionLogger  # noqa: E402
from src.utils import (  # noqa: E402
    file_timestamp,
    hardware_info,
    sanitize_filename,
)
from src.visualization import annotate_image, save_annotated  # noqa: E402
from scripts.predict_video import process_video  # noqa: E402

st.set_page_config(
    page_title="OilBottleVisionDemo - Şişe Kalite Kontrol",
    page_icon="🫒",
    layout="wide",
)

CONFIG = load_config()
CONFIG.ensure_dirs()


# ==========================================================================
# Yardımcılar
# ==========================================================================
@st.cache_resource(show_spinner="Model yükleniyor...")
def get_detector(weights: str | None, device: str | None) -> BottleDetector:
    """Modeli bir kez yükleyip yeniden kullanır."""
    return BottleDetector(
        weights=Path(weights) if weights else None, device=device, config=CONFIG
    )


@st.cache_resource
def get_inspection_logger() -> InspectionLogger:
    return InspectionLogger(CONFIG.inspection_log)


def status_banner(result: InspectionResult) -> None:
    """Kararı renkli bir kutu içinde gösterir."""
    color = result.color_hex
    text_color = "#111111" if result.status_key == STATUS_MANUAL else "#FFFFFF"
    st.markdown(
        f"""
        <div style="background:{color};color:{text_color};padding:18px 22px;
                    border-radius:10px;margin:6px 0 14px 0;">
          <div style="font-size:30px;font-weight:800;letter-spacing:0.5px;">
            {result.status}
          </div>
          <div style="font-size:16px;opacity:0.95;margin-top:4px;">
            {result.reason}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def legend() -> None:
    items = [
        (STATUS_OK, "Sağlam"),
        (STATUS_DEFECT, "Hatalı"),
        (STATUS_WRONG_PRODUCT, "Yanlış ürün"),
        (STATUS_MANUAL, "Manuel kontrol"),
    ]
    chips = "".join(
        f'<span style="background:{STATUS_COLORS_HEX[key]};color:#fff;'
        f'padding:3px 10px;border-radius:12px;margin-right:6px;'
        f'font-size:12px;">{label}</span>'
        for key, label in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def detections_table(result: InspectionResult) -> pd.DataFrame:
    rows = []
    for det in result.detections:
        rows.append(
            {
                "Sınıf": det.class_name,
                "Açıklama": det.label,
                "Güven": f"%{det.confidence * 100:.1f}",
                "Durum": STATUS_LABELS.get(det.rule.status, det.rule.status),
                "Değerlendirme": (
                    "Kararsız (manuel bant)" if det.tier == "uncertain" else "Güvenilir"
                ),
            }
        )
    return pd.DataFrame(rows)


def decode_upload(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ==========================================================================
# Kenar çubuğu - model durumu ve ayarlar
# ==========================================================================
def sidebar() -> tuple[BottleDetector | None, Thresholds]:
    st.sidebar.title("🫒 OilBottleVisionDemo")
    st.sidebar.caption("Yağ şişesi kalite kontrol demosu")

    hw = hardware_info()
    trained_exists = CONFIG.trained_weights.is_file()

    st.sidebar.subheader("Model")
    weight_options: dict[str, str | None] = {}
    if trained_exists:
        weight_options[f"Eğitilmiş model ({CONFIG.trained_weights.name})"] = str(
            CONFIG.trained_weights
        )
    if CONFIG.last_weights.is_file():
        weight_options[f"Son epoch ({CONFIG.last_weights.name})"] = str(
            CONFIG.last_weights
        )
    weight_options[f"Ön eğitimli taban ({CONFIG.model_weights})"] = CONFIG.model_weights

    weight_label = st.sidebar.selectbox("Ağırlık", list(weight_options))
    selected_weights = weight_options[weight_label]

    device_choice = st.sidebar.selectbox(
        "Cihaz",
        ["auto", "0 (GPU)", "cpu"],
        index=0,
        help="auto: GPU varsa kullanılır, yoksa CPU",
    )
    device = {"auto": "auto", "0 (GPU)": "0", "cpu": "cpu"}[device_choice]

    st.sidebar.subheader("Eşikler")
    defaults = CONFIG.thresholds
    conf = st.sidebar.slider(
        "Tespit güven eşiği (detection_confidence)",
        0.05,
        0.95,
        float(defaults.detection_confidence),
        0.01,
    )
    iou = st.sidebar.slider(
        "IoU eşiği", 0.10, 0.90, float(defaults.iou_threshold), 0.01
    )
    review_low, review_high = st.sidebar.slider(
        "Manuel kontrol aralığı",
        0.05,
        0.95,
        (float(defaults.manual_review_low), float(defaults.manual_review_high)),
        0.01,
        help="Bu aralıktaki güven değerleri MANUEL KONTROL kararına yönlendirilir.",
    )
    thresholds = Thresholds(
        detection_confidence=conf,
        manual_review_low=review_low,
        manual_review_high=review_high,
        iou_threshold=iou,
    )
    for warning in thresholds.validate():
        st.sidebar.warning(warning)

    detector: BottleDetector | None = None
    error: str | None = None
    try:
        detector = get_detector(selected_weights, device)
        detector.thresholds = thresholds
        detector.engine.thresholds = thresholds
    except ModelLoadError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - ultralytics çeşitli hatalar atabilir
        error = f"{type(exc).__name__}: {exc}"

    st.sidebar.subheader("Durum")
    if detector is None:
        st.sidebar.error("Model yüklenemedi")
        if error:
            st.sidebar.code(error[:500])
    else:
        info = detector.info()
        st.sidebar.success("Model hazır")
        st.sidebar.markdown(
            f"""
            - **Ağırlık:** `{Path(info['weights']).name}`
            - **Cihaz:** `{info['device']}`
            - **Sınıf sayısı:** {len(info['class_names'])}
            """
        )
        if not info["is_trained"]:
            st.sidebar.warning(
                "Şu anda ön eğitimli taban model kullanılıyor. Gerçek kalite "
                "kontrolü için `python scripts/train.py` ile eğitim yapın."
            )
        if info["unmapped_classes"]:
            st.sidebar.warning(
                "Karar tablosunda karşılığı olmayan sınıflar (MANUEL KONTROL "
                "olarak işlenir): " + ", ".join(info["unmapped_classes"][:8])
            )

    gpu_text = (
        f"{hw['gpus'][0]['name']} ({hw['gpus'][0]['total_memory_gb']} GB)"
        if hw["gpus"]
        else "GPU yok - CPU kullanılıyor"
    )
    st.sidebar.caption(f"Donanım: {gpu_text}")
    st.sidebar.caption(f"PyTorch {hw['torch']} · CUDA {hw['cuda_version'] or '-'}")

    return detector, thresholds


# ==========================================================================
# Sekme 1 - Ana sayfa
# ==========================================================================
def page_home(detector: BottleDetector | None, thresholds: Thresholds) -> None:
    st.header("Ana Sayfa")
    st.write(
        "Bu demo, üretim hattındaki yağ şişelerinin fotoğraf veya video ile "
        "kalite kontrolünü yapar ve her ürünü dört karardan biriyle "
        "sınıflandırır."
    )
    legend()
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    hw = hardware_info()
    with col1:
        st.metric("Proje", "OilBottleVisionDemo")
    with col2:
        st.metric(
            "Model durumu",
            "Hazır" if detector else "Yüklenemedi",
        )
    with col3:
        st.metric("Cihaz", detector.device if detector else "-")
    with col4:
        st.metric("Güven eşiği", f"{thresholds.detection_confidence:.2f}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Model bilgisi")
        if detector:
            info = detector.info()
            st.markdown(
                f"""
                | Alan | Değer |
                |---|---|
                | Kullanılan ağırlık | `{info['weights']}` |
                | Eğitilmiş model mi | {'Evet' if info['is_trained'] else 'Hayır (taban model)'} |
                | Çalışılan cihaz | `{info['device']}` |
                | Tespit eşiği | {info['detection_confidence']:.2f} |
                | IoU eşiği | {info['iou_threshold']:.2f} |
                | Manuel kontrol aralığı | {info['manual_review_low']:.2f} – {info['manual_review_high']:.2f} |
                """
            )
            st.write("**Model sınıfları** (data.yaml'dan gelir):")
            st.code("\n".join(info["class_names"]) or "-")
        else:
            st.error(
                "Model yüklenemedi. Önce eğitim çalıştırın:\n\n"
                "```\npython scripts/prepare_dataset.py\npython scripts/train.py\n```"
            )

    with right:
        st.subheader("Donanım")
        st.markdown(
            f"""
            | Alan | Değer |
            |---|---|
            | Platform | {hw['platform']} |
            | Python | {hw['python']} |
            | PyTorch | {hw['torch'] or '-'} |
            | CUDA | {hw['cuda_version'] or '-'} |
            | GPU kullanımı | {'Evet' if hw['cuda_available'] else 'Hayır (CPU)'} |
            """
        )
        for gpu in hw["gpus"]:
            st.caption(
                f"GPU {gpu['index']}: {gpu['name']} · {gpu['total_memory_gb']} GB"
            )

        st.subheader("Karar kuralları")
        rules = CONFIG.class_rules
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Sınıf": name,
                        "Açıklama": rule.label,
                        "Karar": STATUS_LABELS.get(rule.status, rule.status),
                        "Ciddiyet": rule.severity,
                    }
                    for name, rule in sorted(
                        rules.items(), key=lambda kv: -kv[1].severity
                    )
                ]
            ),
            hide_index=True,
            width="stretch",
        )


# ==========================================================================
# Sekme 2 - Görüntü analizi
# ==========================================================================
def page_image(detector: BottleDetector | None) -> None:
    st.header("Görüntü Analizi")
    if detector is None:
        st.error("Model yüklenemedi; analiz yapılamıyor.")
        return

    uploaded = st.file_uploader(
        "Şişe fotoğrafı yükleyin",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("JPG, JPEG veya PNG formatında bir görüntü yükleyin.")
        return

    frame = decode_upload(uploaded)
    if frame is None:
        st.error("Görüntü okunamadı. Dosya bozuk olabilir.")
        return

    with st.spinner("Analiz ediliyor..."):
        result, _ = detector.predict(frame)
        annotated = annotate_image(frame, result)

    status_banner(result)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal görüntü")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
    with col2:
        st.subheader("Analiz sonucu")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Karar", result.status)
    m2.metric("En yüksek güven", f"%{result.max_confidence * 100:.1f}")
    m3.metric("İşlem süresi", f"{result.inference_time_ms:.0f} ms")

    st.subheader("Bulunan kusurlar")
    if result.defects:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Kusur": d.label,
                        "Sınıf": d.class_name,
                        "Güven": f"%{d.confidence * 100:.1f}",
                    }
                    for d in result.defects
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.success("Kusur tespit edilmedi.")

    if result.uncertain:
        st.warning(
            "Aşağıdaki tespitlerin güven değeri manuel kontrol aralığında; "
            "operatör doğrulaması önerilir."
        )

    with st.expander("Tüm tespitler"):
        table = detections_table(result)
        if table.empty:
            st.write("Tespit yok.")
        else:
            st.dataframe(table, hide_index=True, width="stretch")

    # Kaydetme + log
    stem = sanitize_filename(Path(uploaded.name).stem)
    target_dir = CONFIG.rejected_dir if result.is_defective else CONFIG.predictions_dir
    saved = save_annotated(annotated, target_dir / f"{stem}_{file_timestamp()}.jpg")
    get_inspection_logger().log(
        source_type="image",
        source_name=uploaded.name,
        final_status=result.status,
        defect_classes=result.defect_classes,
        confidence=result.max_confidence,
        inference_time_ms=result.inference_time_ms,
        output_path=saved,
    )
    if result.is_defective:
        st.error(f"Hatalı ürün kaydedildi: `{saved}`")
    else:
        st.caption(f"Sonuç kaydedildi: `{saved}`")

    ok, buffer = cv2.imencode(".jpg", annotated)
    if ok:
        st.download_button(
            "Analiz edilmiş görüntüyü indir",
            data=buffer.tobytes(),
            file_name=f"{stem}_analiz.jpg",
            mime="image/jpeg",
        )


# ==========================================================================
# Sekme 3 - Video analizi
# ==========================================================================
def page_video(detector: BottleDetector | None) -> None:
    st.header("Video Analizi")
    if detector is None:
        st.error("Model yüklenemedi; analiz yapılamıyor.")
        return

    uploaded = st.file_uploader("MP4 video yükleyin", type=["mp4"])
    col_a, col_b, col_c = st.columns(3)
    stride = col_a.number_input(
        "Frame adımı (her N. frame analiz edilir)",
        min_value=1,
        max_value=30,
        value=int(CONFIG.video.get("frame_stride", 3)),
    )
    max_frames = col_b.number_input(
        "En fazla frame (0 = tümü)", min_value=0, max_value=100000, value=0, step=50
    )
    write_video = col_c.checkbox("Analiz videosunu kaydet", value=True)

    if uploaded is None:
        st.info("Üretim hattından alınmış bir MP4 video yükleyin.")
        return

    if not st.button("Analizi başlat", type="primary"):
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = Path(tmp.name)

    stem = sanitize_filename(Path(uploaded.name).stem)
    stamp = file_timestamp()
    output_video = (
        CONFIG.predictions_dir / f"{stem}_{stamp}_analiz.mp4" if write_video else None
    )

    progress = st.progress(0.0, text="Video işleniyor...")

    def on_progress(current: int, total: int) -> None:
        progress.progress(min(1.0, current / max(total, 1)), text=f"Frame {current}/{total}")

    try:
        outcome = process_video(
            temp_path,
            detector,
            CONFIG,
            output_video=output_video,
            stride=int(stride),
            progress_callback=on_progress,
            max_frames=int(max_frames) or None,
        )
    except OSError as exc:
        st.error(f"Video işlenemedi: {exc}")
        return
    finally:
        progress.empty()
        temp_path.unlink(missing_ok=True)

    summary: dict[str, Any] = outcome["summary"]

    st.subheader("Özet")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam incelenen ürün", summary["total_products"])
    c2.metric("Sağlam", summary["ok"])
    c3.metric("Hatalı", summary["defective"])
    c4.metric("Yanlış ürün / Manuel", f"{summary['wrong_product']} / {summary['manual_review']}")

    st.caption(
        f"{summary['total_frames']} frame · {summary['analyzed_frames']} analiz "
        f"· ortalama {summary['avg_inference_ms']:.0f} ms/frame "
        f"· {summary['resolution']} @ {summary['fps']} fps"
    )

    st.subheader("Kusur türlerinin dağılımı")
    dist = summary["defect_distribution"]
    if dist:
        dist_df = pd.DataFrame(
            sorted(dist.items(), key=lambda kv: -kv[1]), columns=["Kusur", "Adet"]
        )
        st.bar_chart(dist_df.set_index("Kusur"))
        st.dataframe(dist_df, hide_index=True, width="stretch")
    else:
        st.info("Kusur tespit edilmedi.")

    st.subheader("Ürün bazlı sonuçlar")
    products = pd.DataFrame(outcome["products"])
    if not products.empty:
        st.dataframe(products, hide_index=True, width="stretch")
        st.download_button(
            "Ürün listesini CSV indir",
            data=products.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stem}_urunler.csv",
            mime="text/csv",
        )

    if summary["output_video"] and Path(summary["output_video"]).is_file():
        st.subheader("Analiz edilmiş video")
        st.caption(f"Kaydedildi: `{summary['output_video']}`")
        st.video(str(summary["output_video"]))

    if summary["rejected_frames"]:
        st.subheader("Hatalı ürün kareleri")
        st.caption(f"{len(summary['rejected_frames'])} kare `{CONFIG.rejected_dir}` altına kaydedildi.")
        for path in summary["rejected_frames"][:8]:
            st.image(path, width=320)

    logger = get_inspection_logger()
    for product in outcome["products"]:
        logger.log(
            source_type="video",
            source_name=f"{uploaded.name}#urun{product['product_id']}",
            final_status=product["status"],
            defect_classes=product["defect_classes"],
            confidence=product["max_confidence"],
            inference_time_ms=summary["avg_inference_ms"],
            output_path=summary["output_video"] or "",
        )


# ==========================================================================
# Sekme 4 - Geçmiş kayıtlar
# ==========================================================================
def page_history() -> None:
    st.header("Geçmiş Kayıtlar")
    st.caption(f"Kayıt dosyası: `{CONFIG.inspection_log}`")

    df = get_inspection_logger().read()
    if df.empty:
        st.info("Henüz kayıt yok. Görüntü veya video analizi yapın.")
        return

    display = df.rename(
        columns={
            "timestamp": "Tarih/Saat",
            "source_type": "Kaynak türü",
            "source_name": "Dosya adı",
            "final_status": "Nihai karar",
            "defect_classes": "Kusur türleri",
            "confidence": "En yüksek güven",
            "inference_time_ms": "İşlem süresi (ms)",
            "output_path": "Çıktı yolu",
        }
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        statuses = sorted(display["Nihai karar"].dropna().unique())
        selected = st.multiselect("Karara göre filtrele", statuses, default=statuses)
    with col2:
        source_types = sorted(display["Kaynak türü"].dropna().unique())
        selected_sources = st.multiselect(
            "Kaynak türü", source_types, default=source_types
        )

    filtered = display[
        display["Nihai karar"].isin(selected)
        & display["Kaynak türü"].isin(selected_sources)
    ]

    counts = display["Nihai karar"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam kontrol", len(display))
    c2.metric("Sağlam", int(counts.get("SAĞLAM", 0)))
    c3.metric("Hatalı", int(counts.get("HATALI", 0)))
    c4.metric(
        "Manuel / Yanlış ürün",
        f"{int(counts.get('MANUEL KONTROL GEREKLİ', 0))} / "
        f"{int(counts.get('YANLIŞ ÜRÜN', 0))}",
    )

    st.dataframe(
        filtered.sort_values("Tarih/Saat", ascending=False),
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Kayıtları CSV olarak indir",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"inspections_{file_timestamp()}.csv",
        mime="text/csv",
    )


# ==========================================================================
def main() -> None:
    detector, thresholds = sidebar()
    tabs = st.tabs(
        ["🏠 Ana Sayfa", "🖼️ Görüntü Analizi", "🎬 Video Analizi", "📋 Geçmiş Kayıtlar"]
    )
    with tabs[0]:
        page_home(detector, thresholds)
    with tabs[1]:
        page_image(detector)
    with tabs[2]:
        page_video(detector)
    with tabs[3]:
        page_history()


main()
