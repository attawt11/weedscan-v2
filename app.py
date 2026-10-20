"""
WeedScan v2 — Weed Detection Demo (Baseline Model)
โมเดลพื้นฐาน weed_only_v2 (YOLOv8s, เทรนจากชุดข้อมูลสาธารณะ, imgsz=640)
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ─── Page config ───
st.set_page_config(
    page_title="WeedScan v2 — Baseline Demo",
    page_icon="🌾",
    layout="wide",
)

# ─── Styling ───
st.markdown("""
<style>
    .main-title { text-align: center; color: #2E7D32; margin-bottom: 0; }
    .sub-title  { text-align: center; color: #666; font-size: 1.1rem; margin-top: 0; }
    .metric-box {
        background: #f0f7f0; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #c8e6c9;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #2E7D32; }
    .metric-lbl { font-size: 0.85rem; color: #666; }
    .version-badge {
        background: #fff3e0; color: #e65100; padding: 4px 12px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        display: inline-block; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<h1 class="main-title">🌾 WeedScan v2</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ระบบตรวจจับวัชพืชในนาข้าว — โมเดลพื้นฐาน (Baseline)</p>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center"><span class="version-badge">'
    'weed_only_v2 · YOLOv8s · imgsz 640 · Public Dataset Only'
    '</span></div>',
    unsafe_allow_html=True,
)

# ─── Load model ───
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"ไม่พบไฟล์โมเดล `{MODEL_PATH}` — กรุณาวางไฟล์ best.pt ของ weed_only_v2 ในโฟลเดอร์เดียวกับ app.py")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    show_labels = st.checkbox("แสดง label", value=True)
    show_conf = st.checkbox("แสดง confidence", value=True)

    st.divider()
    st.markdown("### 📊 ข้อมูลโมเดล")
    st.markdown("""
    | รายการ | ค่า |
    |---|---|
    | สถาปัตยกรรม | YOLOv8s (11.1M params) |
    | ขนาดภาพ | 640×640 |
    | ชุดข้อมูล | สาธารณะเท่านั้น |
    | Transfer Learning | COCO → weed-only+BG |
    | SAHI | ❌ ไม่ใช้ |
    """)

    st.divider()
    st.caption("🔬 โมเดลพื้นฐานสำหรับเปรียบเทียบ")
    st.caption("เทรนจากชุดข้อมูลสาธารณะ ~5,845 ภาพ")

# ─── Upload ───
st.markdown("---")
uploaded = st.file_uploader(
    "📷 อัปโหลดภาพนาข้าว",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
    help="รองรับ JPG, PNG, WebP, BMP",
)

if not uploaded:
    # Show sample info
    st.info("⬆️ อัปโหลดภาพเพื่อเริ่มตรวจจับวัชพืช — รองรับหลายภาพพร้อมกัน")
    st.stop()

# ─── Process each image ───
for file in uploaded:
    st.markdown(f"### 📸 {file.name}")

    image = Image.open(file).convert("RGB")
    img_np = np.array(image)

    # Run YOLOv8 inference (v2 baseline — no SAHI, imgsz=640)
    results = model.predict(
        img_np,
        imgsz=640,
        conf=conf_threshold,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    # Draw results
    annotated = result.plot(
        labels=show_labels,
        conf=show_conf,
        line_width=2,
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if annotated.shape[2] == 3 else annotated

    # Layout: image + metrics
    col_img, col_info = st.columns([3, 1])

    with col_img:
        tab_det, tab_orig = st.tabs(["🔍 ผลตรวจจับ", "🖼️ ภาพต้นฉบับ"])
        with tab_det:
            st.image(annotated_rgb, use_container_width=True)
        with tab_orig:
            st.image(image, use_container_width=True)

    with col_info:
        n_weeds = len(boxes)
        avg_conf = float(boxes.conf.mean()) * 100 if n_weeds > 0 else 0

        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{n_weeds}</div>'
            f'<div class="metric-lbl">วัชพืชที่พบ</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{avg_conf:.1f}%</div>'
            f'<div class="metric-lbl">ความมั่นใจเฉลี่ย</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{image.size[0]}×{image.size[1]}</div>'
            f'<div class="metric-lbl">ขนาดภาพ (px)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Download annotated image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(annotated_rgb).save(tmp.name, quality=95)
            with open(tmp.name, "rb") as f:
                st.download_button(
                    "💾 ดาวน์โหลดผลลัพธ์",
                    f.read(),
                    file_name=f"v2_detected_{file.name}",
                    mime="image/jpeg",
                    use_container_width=True,
                )

    # Detail table
    if n_weeds > 0:
        with st.expander(f"📋 รายละเอียด ({n_weeds} กล่อง)"):
            import pandas as pd
            data = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                data.append({
                    "#": i + 1,
                    "Confidence": f"{float(box.conf):.3f}",
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "Width": int(x2 - x1),
                    "Height": int(y2 - y1),
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.divider()

# ─── Footer ───
st.markdown(
    '<div style="text-align:center; color:#999; font-size:0.8rem; padding:20px 0">'
    'WeedScan v2 (Baseline) — ระบบตรวจจับวัชพืชในนาข้าวด้วย Deep Learning<br>'
    'YOLOv8s · weed_only_v2 · KMITL'
    '</div>',
    unsafe_allow_html=True,
)
