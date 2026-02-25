import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO
from pathlib import Path
from collections import Counter

# ==============================
# PATH SETUP
# ==============================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ==============================
# MODEL LOADING (CACHED)
# ==============================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

@st.cache_resource
def get_annotators():
    return sv.BoxAnnotator(), sv.LabelAnnotator()

# ==============================
# DETECTION PIPELINE
# ==============================
def detector_pipeline_pillow(image_bytes, model, conf=0.5):

    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)

    results = model(image_np_rgb, conf=conf, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    box_annotator, label_annotator = get_annotators()

    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    annotated_image_np = np.asarray(annotated_image)

    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))

    return annotated_image_np, classcounts


# ==============================
# ANALYSIS LOGIC
# ==============================
def analyze_construction_safety(classcounts):

    required_classes = ["person", "helmet", "vest", "no-helmet", "no-vest"]

    # pastikan semua key ada
    for cls in required_classes:
        classcounts.setdefault(cls, 0)

    total_person = classcounts["person"]
    no_helmet = classcounts["no-helmet"]
    no_vest = classcounts["no-vest"]

    violations = no_helmet + no_vest

    if total_person > 0:
        compliance = ((total_person - violations) / total_person) * 100
    else:
        compliance = 0

    # Risk Level
    if compliance >= 90:
        risk = "Low"
    elif compliance >= 70:
        risk = "Medium"
    else:
        risk = "High"

    # Insight otomatis
    insight = (
        f"Dari {total_person} pekerja terdeteksi, "
        f"terdapat {violations} pelanggaran keselamatan. "
        f"Tingkat kepatuhan sebesar {compliance:.2f}%. "
        f"Kategori risiko: {risk}."
    )

    return compliance, risk, insight


# ==============================
# STREAMLIT APP
# ==============================

st.set_page_config(page_title="Construction Safety Detection", layout="wide")
st.title("🏗 Construction Safety Monitoring System")

selected_model = st.selectbox("Select Usecase", ("Construction Equipment", "Vehicle", "Fruit"))

model_map = {
    "Construction Equipment": MODELS_DIR / "best_construction.pt",
    "Vehicle": MODELS_DIR / "best_vehicle.pt",
    "Fruit": MODELS_DIR / "best_fruit.pt"
}

with st.spinner(f"Loading {selected_model} model..."):
    model = load_model(model_map[selected_model])

st.success(f"✅ {selected_model} model loaded!")


# Confidence slider (nilai tambah kompleksitas)
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

uploaded_file = st.file_uploader(
    "Upload Image",
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Objects", type="primary"):

        bytes_data = uploaded_file.getvalue()

        with st.spinner("Detecting objects..."):
            annotated_image_rgb, classcounts = detector_pipeline_pillow(
                bytes_data,
                model,
                conf=conf_threshold
            )

        st.subheader("📸 Detection Result")
        st.image(annotated_image_rgb, use_container_width=True)

        if classcounts:

            st.subheader("📊 Object Counts")
            col1, col2 = st.columns([1, 2])

            with col1:
                for class_name, count in classcounts.items():
                    st.metric(label=class_name, value=count)

            # ==========================
            # SAFETY ANALYSIS SECTION
            # ==========================
            if selected_model == "Construction Equipment":

                st.subheader("🛡 Safety Compliance Analysis")

                compliance, risk, insight = analyze_construction_safety(classcounts)

                st.metric("Compliance Rate", f"{compliance:.2f}%")

                if risk == "High":
                    st.error(f"Risk Level: {risk}")
                elif risk == "Medium":
                    st.warning(f"Risk Level: {risk}")
                else:
                    st.success(f"Risk Level: {risk}")

                st.info(insight)

        else:
            st.info("No objects detected in the image.")