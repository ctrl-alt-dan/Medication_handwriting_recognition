import streamlit as st
from PIL import Image

from utils import (
    read_pil_from_streamlit_file,
    apply_rotate,
    draw_crop_guidance_overlay,
    preprocess_for_model,
    load_labels,
    format_topk,
)
from model import load_model, predict_topk


# -----------------------------
# Page config (mobile friendly)
# -----------------------------
st.set_page_config(
    page_title="Handwritten Medication Predictor",
    page_icon="💊",
    layout="centered",
)

st.title("💊 Handwritten Medication Predictor")
st.caption("Take a photo or upload an image, crop/rotate it, then get Top‑5 predictions.")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- Works with **phone camera** or **camera roll** images.
- You can **rotate** (fix tilt) and **crop** (focus on the medication word).
- The app runs your trained CNN and returns the **Top‑5 medication names** with probabilities.
        """.strip()
    )

# -----------------------------
# Sidebar: model files
# -----------------------------
st.sidebar.header("Model files")
weights_path = st.sidebar.text_input("Weights (.pt)", value="best_tuned_cnn.pt")
labels_path  = st.sidebar.text_input("Labels (classes.json or idx_to_class.json)", value="classes.json")
top_k        = st.sidebar.slider("Top‑K outputs", min_value=1, max_value=10, value=5, step=1)

st.sidebar.divider()
st.sidebar.caption("Tip: keep the word large and centered. Avoid glare/shadows.")

# -----------------------------
# Load model + labels (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_model_cached(weights_path: str):
    return load_model(weights_path)

@st.cache_data(show_spinner=False)
def _load_labels_cached(labels_path: str):
    return load_labels(labels_path)

try:
    model, device = _load_model_cached(weights_path)
except Exception as e:
    st.error(f"Could not load model weights from: {weights_path}\n\n{e}")
    st.stop()

try:
    labels = _load_labels_cached(labels_path)
except Exception as e:
    st.error(f"Could not load labels from: {labels_path}\n\n{e}")
    st.stop()

# -----------------------------
# Inputs: camera + upload
# -----------------------------
st.subheader("1) Provide an image")

colA, colB = st.columns(2, gap="small")

with colA:
    cam_file = st.camera_input("Take a photo")

with colB:
    up_file = st.file_uploader(
        "Upload from camera roll",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False
    )

file = cam_file if cam_file is not None else up_file
if file is None:
    st.info("Upload an image or take a photo to begin.")
    st.stop()

# Read to PIL (handles EXIF rotation from phones)
pil = read_pil_from_streamlit_file(file)

st.subheader("2) Rotate & crop (optional but recommended)")

# Rotate helper
rot_deg = st.slider("Rotate (degrees)", min_value=-30, max_value=30, value=0, step=1)
pil_rot = apply_rotate(pil, rot_deg)

# Crop helper (uses streamlit-cropper)
from streamlit_cropper import st_cropper

st.caption("Crop to just the medication word. A wide aspect ratio works best.")


show_guide = st.checkbox("Show crop guidance overlay", value=True)
if show_guide:
    guided = draw_crop_guidance_overlay(pil_rot, target_aspect=(384/64))
    st.image(guided, caption="Guide: keep the medication word inside the rectangle", use_container_width=True)

cropped = st_cropper(
    pil_rot,
    realtime_update=True,
    aspect_ratio=(384 / 64),  # match model input ratio
    box_color="#00BFFF",
    return_type="PIL",
)

# Preview
st.markdown("**Preview**")
st.image(cropped, use_container_width=True)

# -----------------------------
# Run prediction
# -----------------------------
st.subheader("3) Predict")

predict_btn = st.button("🔎 Run prediction", use_container_width=True)

if predict_btn:
    with st.spinner("Preprocessing + predicting..."):
        x = preprocess_for_model(cropped)  # torch tensor [1,1,64,384]
        top = predict_topk(model, x, top_k=top_k, device=device)

    # Display results
    st.success("Done!")
    results = format_topk(top, labels)

    st.markdown("### Top predictions")
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("### Probability bars")
    for _, row in results.iterrows():
        st.write(f"**{row['medication']}** — {row['probability_pct']}")
        st.progress(min(max(row["probability_float"], 0.0), 1.0))

    # Optional: show what the model sees
    with st.expander("Show preprocessed model input (64×384)", expanded=False):
        from utils import preprocess_debug_image
        dbg = preprocess_debug_image(cropped)
        st.image(dbg, caption="Preprocessed (grayscale / enhanced / resized & padded)", use_container_width=True)
