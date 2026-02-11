'''
Streamlit Application: Batch Image Segmentation and Fluorescent Counting Model
'''

import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
import seaborn as sns
from PIL import Image
from scipy.stats import mannwhitneyu
from skimage.measure import label, regionprops
import pandas as pd
from pathlib import Path
import cv2
import gc
import zipfile

# ────────────────────────────────────────────────
# Base directory (works locally + on cloud)
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ────────────────────────────────────────────────
# Clear memory
# ────────────────────────────────────────────────
tf.keras.backend.clear_session()
gc.collect()

# ────────────────────────────────────────────────
# Load CSS safely
# ────────────────────────────────────────────────
def load_css():
    css_path = BASE_DIR / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("styles.css not found — custom styles skipped.")

load_css()

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Batch Image Segmentation and Fluorescent Counting Model",
    layout="wide"
)

st.header(":blue[Batch Image Segmentation and Fluorescent Counting Model]", divider="green")

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.success(f"GPU Available: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        st.error(f"GPU config error: {e}")
else:
    st.warning("No GPU detected — running on CPU.")

# ────────────────────────────────────────────────
# Custom objects
# ────────────────────────────────────────────────
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return tf.pow(1 - tversky, gamma)

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

custom_objects = {
    "tversky_loss": tversky_loss,
    "dice_coef": dice_coef,
    "jacard": jacard,
    "combined_loss": combined_loss,
    "dice_loss": dice_loss,
}

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = BASE_DIR / "models" / "SDU_best_fold_1.h5"
    
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.error("Upload SDU_best_fold_1.h5 to the 'models' folder in GitHub.")
        st.stop()

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error("Failed to load model.")
        st.exception(e)
        st.stop()

model = load_model()

# ────────────────────────────────────────────────
# Core functions
# ────────────────────────────────────────────────
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def count_cells(image, model, crop_params):
    image = np.array(image)
    x, y, w, h = crop_params
    image = crop_image(image, x, y, w, h)
    h_img, w_img = image.shape[:2]
    patch_size = 128

    pad_h = (patch_size - h_img % patch_size) % patch_size
    pad_w = (patch_size - w_img % patch_size) % patch_size
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    H, W, _ = padded_image.shape
    predicted_mask = np.zeros((H, W), dtype=np.uint8)

    patches = []
    coords = []

    for y_patch in range(0, H, patch_size):
        for x_patch in range(0, W, patch_size):
            patch = padded_image[y_patch:y_patch+patch_size, x_patch:x_patch+patch_size]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) / 255.0
            patches.append(patch)
            coords.append((x_patch, y_patch))

    if not patches:
        return 0, np.zeros((h_img, w_img), dtype=np.uint8), 0.0, []

    patches = np.array(patches)
    batch_size = 8

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        preds = (preds > 0.5).astype(np.uint8)
        for j, pred in enumerate(preds):
            xp, yp = coords[i + j]
            predicted_mask[yp:yp+patch_size, xp:xp+patch_size] = pred.squeeze()

    predicted_mask = predicted_mask[:h_img, :w_img]

    labeled = label(predicted_mask)
    props = regionprops(labeled)
    areas = [round(float(p.area), 4) for p in props if p.area > 0]
    total_area = round(float(sum(areas)), 10)

    return len(props), predicted_mask, total_area, areas

def classify_image(cell_count, threshold):
    return "High Density" if cell_count > threshold else "Low Density"

def save_image(image, filename, save_dir):
    image.save(os.path.join(save_dir, filename))

def save_mask(prediction, filename, save_dir):
    Image.fromarray((prediction * 255).astype(np.uint8)).save(os.path.join(save_dir, filename))

def process_and_save(images, label, threshold, save_dir=None):
    st.subheader(f"Results for {label}")
    crop_params = crop_settings.get(label, (0, 0, 512, 512))

    for idx, uploaded_file in enumerate(images):
        image = Image.open(uploaded_file)
        cell_count, prediction, total_area, areas = count_cells(image, model, crop_params)

        # Safely extend areas (already guaranteed to exist)
        areas_groups[label].extend(areas)

        image_results.append({
            "Image Name": uploaded_file.name,
            "Group": label,
            "Particle Count": cell_count,
            "Total Area": total_area
        })

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
            if save_dir:
                save_image(image, f"{label}_original_{idx+1}.png", save_dir)

        with col2:
            st.markdown("**Predicted Mask**")
            fig, ax = plt.subplots()
            cmap = 'gray' if len(np.unique(prediction)) <= 2 else 'viridis'
            ax.imshow(prediction, cmap=cmap)
            ax.axis('off')
            st.pyplot(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{label}_mask_{idx+1}.png", bbox_inches='tight')
            plt.close(fig)

        with col3:
            st.markdown(f"""
            **Statistics**  
            Fluorescent Signals: **{cell_count}**  
            Total Area: **{total_area} px²**
            """)

def create_save_directory():
    save_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def zip_dir(directory_path):
    zip_path = f"{directory_path}.zip"
    shutil.make_archive(directory_path, 'zip', directory_path)
    return zip_path

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'image_results' not in st.session_state:
    st.session_state['image_results'] = []
if 'areas_groups' not in st.session_state:
    st.session_state['areas_groups'] = {}
if 'save_dir' not in st.session_state:
    st.session_state['save_dir'] = None
if 'evaluation_triggered' not in st.session_state:
    st.session_state['evaluation_triggered'] = False
if 'download_triggered' not in st.session_state:
    st.session_state['download_triggered'] = False

image_results = st.session_state['image_results']
areas_groups = st.session_state['areas_groups']

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.sidebar.markdown("### Settings")
threshold = st.sidebar.slider("Fluorescent Count Threshold", 10, 100, 30)
concentration = st.sidebar.slider("Concentration Used (ng)", 0, 100, 30)

st.markdown("### Group Labels")
col1, col2 = st.columns(2)
with col1:
    label_g1 = st.text_input("Group 1", "Control")
with col2:
    label_g2 = st.text_input("Group 2", "Patient")

col3, col4 = st.columns(2)
with col3:
    label_g3 = st.text_input("Group 3", "Image Set 3")
with col4:
    label_g4 = st.text_input("Group 4", "Image Set 4")

# IMPORTANT: Dynamically ensure every current label has a list
for lbl in [label_g1, label_g2, label_g3, label_g4]:
    if lbl and lbl not in areas_groups:
        areas_groups[lbl] = []

st.markdown("### Upload Images")
col5, col6 = st.columns(2)
with col5:
    files_g1 = st.file_uploader("Group 1", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])
with col6:
    files_g2 = st.file_uploader("Group 2", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

col7, col8 = st.columns(2)
with col7:
    files_g3 = st.file_uploader("Group 3", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])
with col8:
    files_g4 = st.file_uploader("Group 4", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

# Crop settings
st.sidebar.markdown("### Crop Settings")
crop_settings = {}
for lbl in [label_g1, label_g2, label_g3, label_g4]:
    if lbl:
        st.sidebar.subheader(lbl)
        x = st.sidebar.number_input(f"{lbl} x", 0, 4000, 15)
        y = st.sidebar.number_input(f"{lbl} y", 0, 4000, 48)
        w = st.sidebar.number_input(f"{lbl} width", 10, 5000, 2529)
        h = st.sidebar.number_input(f"{lbl} height", 10, 4000, 1947)
        crop_settings[lbl] = (x, y, w, h)

# ────────────────────────────────────────────────
# Run Processing
# ────────────────────────────────────────────────
if st.button("Run Processing"):
    st.session_state['evaluation_triggered'] = True
    save_dir = create_save_directory()
    st.session_state['save_dir'] = save_dir

    if any([files_g1, files_g2, files_g3, files_g4]):
        with st.spinner("Processing images..."):
            if files_g1:
                process_and_save(files_g1, label_g1, threshold, save_dir)
            if files_g2:
                process_and_save(files_g2, label_g2, threshold, save_dir)
            if files_g3:
                process_and_save(files_g3, label_g3, threshold, save_dir)
            if files_g4:
                process_and_save(files_g4, label_g4, threshold, save_dir)

        st.success("Processing completed!")
    else:
        st.warning("Upload at least one group of images.")

# ────────────────────────────────────────────────
# Download Results
# ────────────────────────────────────────────────
if st.sidebar.button("Download Results"):
    if st.session_state.get('save_dir') and st.session_state.get('evaluation_triggered'):
        with st.spinner("Creating ZIP..."):
            zip_path = zip_dir(st.session_state['save_dir'])
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download results.zip",
                    f,
                    file_name="results.zip",
                    mime="application/zip"
                )
    else:
        st.warning("Run processing first.")