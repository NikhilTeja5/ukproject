'''
Streamlit Application:
15/03 -  Bioiminage_6 copied and arranging
'''
# ''' Importing Pkgs '''
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
from skimage.measure import label,regionprops
import pandas as pd
from keras import backend as K
from tensorflow.keras import backend as K
import cv2
import keras.losses
import zipfile
import gc
import collections

# Clear TensorFlow session and free memory
K.clear_session()
gc.collect()        # run Python garbage collector

# Load the CSS file
with open("styles.css") as f:
    css = f.read()

# Cleanup logic: delete old save_dir on rerun unless new evaluation triggered
if 'save_dir' in st.session_state:
    save_dir = st.session_state['save_dir']
    if save_dir and isinstance(save_dir, str) and os.path.exists(save_dir):
        if not st.session_state.get('evaluation_triggered', False) and not st.session_state.get('download_triggered', False):
            shutil.rmtree(save_dir, ignore_errors=True)
            del st.session_state['save_dir']

st.set_page_config(page_title="Batch Image Segmentation and Fluorescent Counting Model", layout="wide")
st.header(":blue[Batch Image Segmentation and Fluorescent Counting Model]", divider="green")
# Inject CSS into Streamlit app
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# ✅ GPU check here
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.success(f"✅ GPU Available: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        st.error(f"RuntimeError during GPU configuration: {e}")
else:
    st.warning("⚠️ No GPU detected. Running on CPU.")
# ''' Importing Custom Modules '''
# Define your custom loss function
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return K.pow(1 - tversky, gamma)
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
def jacard(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten labels
    y_pred_f = K.flatten(y_pred)  # Flatten predictions
    intersection = K.sum(y_true_f * y_pred_f)  # Compute intersection
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def combined_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)
def tversky_losses(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7):
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = numerator + alpha * tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3]) + beta * tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    return 1 - numerator / denominator
custom_objects = {
    "tversky_loss": tversky_loss,
    "dice_coef": dice_coef,
    "jacard": jacard,
    "combined_loss":combined_loss
}
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fixed

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]
def count_cells(image, model, crop_params ):

    image = np.array(image)
    x, y, w, h = crop_params

    # Crop to region of interest
    image = image[y:y+h, x:x+w]
    h, w = image.shape[:2]
    patch_size=128

    # Padding to make divisible by patch size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )

    H, W, _ = padded_image.shape
    predicted_mask = np.zeros((H, W), dtype=np.uint8)

    patches = []
    coords = []

    # Preprocess all patches and store their coordinates
    for y_patch in range(0, H, patch_size):
        for x_patch in range(0, W, patch_size):
            patch = padded_image[y_patch:y_patch+patch_size, x_patch:x_patch+patch_size]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) / 255.0
            patches.append(patch)
            coords.append((x_patch, y_patch))

    # Predict in batches for speed
    patches = np.array(patches)
    batch_size = 8
    with tf.device('/GPU:0'):
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            preds = model.predict(batch, verbose=0)
            preds = (preds > 0.5).astype(np.uint8)

            for j, pred in enumerate(preds):
                x_patch, y_patch = coords[i + j]
                predicted_mask[y_patch:y_patch+patch_size, x_patch:x_patch+patch_size] = pred.squeeze()

    # Crop back to original size
    predicted_mask = predicted_mask[:h, :w].astype(np.uint8)

    # Post-processing
    labeled_image = label(predicted_mask)
    props = regionprops(labeled_image)
    areas = [round(float(prop.area), 4) for prop in props if prop.area > 0]
    total_area = round(float(sum(areas)), 10)

    return len(props), predicted_mask, total_area, areas
def classify_image(cell_count, threshold):
    return "High Density" if cell_count > threshold else "Low Density"
def save_image(image, filename, save_dir):
    image.save(os.path.join(save_dir, filename))
def save_mask(prediction, filename, save_dir):
    mask_image = Image.fromarray((prediction * 255).astype(np.uint8))
    mask_image.save(os.path.join(save_dir, filename))
def process_and_save(images, label, threshold, save_dir=None):
    st.write(f"### Results for {label}:")
    crop_params = crop_settings[label]  # <-- get crop for the current label

    for idx, image_file in enumerate(images):
        image = Image.open(image_file)
        cell_count, prediction, total_area, areas = count_cells(image, model=model, crop_params=crop_params)
        # Store areas for group
        areas_groups[label].extend(areas)
                # Store image name, count, and total area
        image_results.append({
            "Image Name": image_file.name,
            "Group": label,
            "Particle Count": cell_count,
            "Total Area": total_area
        })
        # Display results
        classification = classify_image(cell_count, threshold)
        # Columns inside the card
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Original Image")
            st.image(image, caption=f"**{label}**", use_container_width=True)
            if save_dir:
                save_image(image, f"{label}_original_{idx + 1}.png", save_dir)

        with col2:
            st.markdown("#### Predicted Mask")
            fig, ax = plt.subplots()
            cmap = 'gray' if np.unique(prediction).tolist() in ([0, 1], [0], [1]) else 'viridis'
            im = ax.imshow(prediction, cmap=cmap)
            ax.axis("off")
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{label}_mask_colorbar_{idx + 1}.png", bbox_inches='tight')

        with col3:
            stats_html = f"""
                <div class='stats'style='font-size: 22px; font-weight: 500;'>
                    <p><strong>📊 Image Statistics</strong></p>
                    <p>🔬 <strong>Fluorescent Signals:</strong> <code>{cell_count}</code></p>
                    <p>🧪 <strong>Total Area (Pixel<sup>2</sup>):</strong> <code>{total_area}</code></p>
                </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)
def calculate_histogram_statistics(data, bins=30):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_count = np.sum(hist)
    mean = np.sum(hist * bin_centers) / total_count
    mean = round(mean, 10)
    cumulative_hist = np.cumsum(hist)
    median_bin = np.where(cumulative_hist >= total_count / 2)[0][0]
    median = bin_centers[median_bin]
    median = round(median, 10)
    variance = np.sum(hist * (bin_centers - mean) ** 2) / total_count
    std_dev = np.sqrt(variance)
    std_dev = round(std_dev, 10)
    return mean, median, std_dev
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]
def create_save_directory():
    save_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
def zip_dir(directory_path):
    zip_path = f"{directory_path}.zip"
    shutil.make_archive(directory_path, 'zip', directory_path)
    return zip_path
def save_plot(fig, filename, save_dir):
    fig.savefig(os.path.join(save_dir, filename))
def mann_whitney_test(group1, group2, label1, label2):
    """ Perform Mann-Whitney U test between two groups and return p-value """
    if not group1 or not group2:
        return None  # Skip test if a group has no data
    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    return stat, p_value
def add_significance_annotations(ax, data_groups, group_labels, alpha_levels=(0.05, 0.01, 0.001)): 
    """
    Adds significance annotations to a boxplot using Mann-Whitney U Test.

    Parameters:
    - ax: matplotlib axis object to annotate.
    - data_groups: list of data arrays, one per group.
    - group_labels: list of labels corresponding to each group.
    - alpha_levels: tuple of thresholds for *, **, *** (default: 0.05, 0.01, 0.001)
    """
    y_max = max(max(group) for group in data_groups)
    y_min = min(min(group) for group in data_groups)
    y_range = y_max - y_min

    base_y = y_max + 0.02 * y_range  # Slightly above the top
    step = 0.05 * y_range            # Vertical space between lines
    line_height = 0.015 * y_range    # Height of the horizontal annotation line
    text_offset = 0.01 * y_range     # Offset for the stars

    occupied = []

    for i in range(len(group_labels)):
        for j in range(i + 1, len(group_labels)):
            data1 = data_groups[i]
            data2 = data_groups[j]
            stat, p_value = mannwhitneyu(data1, data2, alternative="two-sided")

            # Determine significance stars
            if p_value < alpha_levels[2]:
                significance = "***"
            elif p_value < alpha_levels[1]:
                significance = "**"
            elif p_value < alpha_levels[0]:
                significance = "*"
            else:
                continue  # Not significant

            x1, x2 = i, j
            y = base_y

            # Avoid overlapping with previous annotations
            while any(x1 <= ox2 and x2 >= ox1 and abs(y - oy) < step for ox1, ox2, oy in occupied):
                y += step

            occupied.append((x1, x2, y))

            # Draw connecting lines
            ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1.5, c='black')
            ax.text((x1 + x2) * 0.5, y + line_height + text_offset, significance,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Extend y-axis limit to fit annotations
    if occupied:
        max_y = max(y for _, _, y in occupied)
        ax.set_ylim(top=max_y + step * 2)
if 'image_results' not in st.session_state:
    st.session_state['image_results'] = []

if 'areas_groups' not in st.session_state:
    st.session_state['areas_groups'] = {}

if 'fig_hist' not in st.session_state:
    st.session_state['fig_hist'] = None

if 'fig_box' not in st.session_state:
    st.session_state['fig_box'] = None

if 'save_dir' not in st.session_state:
    st.session_state['save_dir'] = None

if 'p_value' not in st.session_state:
    st.session_state['p_value'] = {}


####UI design

# model=tf.keras.models.load_model("/mnt/d/Academics/IIT Hyderabad/Project/2025_Project_Work/Weights/IL6_50/Simplified_Depthwise_Unet_with_residual/128_size/SDU_900_bce_dice_8_small_dset_50/train_result/SDU_best_fold_1.h5",custom_objects=custom_objects)
# model=tf.keras.models.load_model("/mnt/d/Academics/IIT Hyderabad/Project/2025_Project_Work/Weights/saved_model_after_900_epochs_depth_512 - Copy.h5",custom_objects=custom_objects)
model=tf.keras.models.load_model("C:\\Users\\Bioimaging\\Desktop\\UK_Project\\SDU_best_fold_1.h5",custom_objects=custom_objects)
threshold = st.sidebar.slider("Fluorescent Count Threshold", min_value=10, max_value=100, value=30)
Concentration_used = st.sidebar.slider("Concentration Used (Nanogram)", min_value=0, max_value=100, value=30)

st.markdown("📝 Enter Group Labels", unsafe_allow_html=True)

# Row 1: Labels for Group 1 and 2
col1, col2 = st.columns(2)
with col1:
    
    label_group_1 = st.text_input("🟦 Group 1 Label", "Control")
with col2:
    label_group_2 = st.text_input("🟨 Group 2 Label", "Patient")

# Row 2: Labels for Group 3 and 4
col3, col4 = st.columns(2)
with col3:
    label_group_3 = st.text_input("🟥 Group 3 Label", "Image Set 3")
with col4:
    label_group_4 = st.text_input("🟪 Group 4 Label", "Image Set 4")

# Colorful divider
st.markdown("---")
st.markdown("📤 Upload Group Images", unsafe_allow_html=True)
# Row 3: Uploaders for Group 1 and 2
col5, col6 = st.columns(2)
with col5:
    uploaded_files_group_1 = st.file_uploader("📁 Upload for Group 1", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])
with col6:
    uploaded_files_group_2 = st.file_uploader("📁 Upload for Group 2", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

# Row 4: Uploaders for Group 3 and 4
col7, col8 = st.columns(2)
with col7:
    uploaded_files_group_3 = st.file_uploader("📁 Upload for Group 3", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])
with col8:
    uploaded_files_group_4 = st.file_uploader("📁 Upload for Group 4", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

image_results = []
# Initialize area data storage for all groups
areas_groups = {
    label_group_1: [],
    label_group_2: [],
    label_group_3: [],
    label_group_4: []
}
# Sidebar inputs for cropping each group
st.sidebar.markdown("### Crop Settings for Each Group")

crop_settings = {}

for group_label in [label_group_1, label_group_2, label_group_3, label_group_4]:
    st.sidebar.markdown(f"**{group_label}**")
    #  x, y, w, h = 1, 1, 512, 420
    x = st.sidebar.number_input(f"{group_label} - Crop x", min_value=0, value=15, step=1)
    y = st.sidebar.number_input(f"{group_label} - Crop y", min_value=0, value=48, step=1)
    w = st.sidebar.number_input(f"{group_label} - Crop width", min_value=10, value=2529, step=1)
    h = st.sidebar.number_input(f"{group_label} - Crop height", min_value=10, value=1947, step=1)
    # x = st.sidebar.number_input(f"{group_label} - Crop x", min_value=0, value=1, step=1)
    # y = st.sidebar.number_input(f"{group_label} - Crop y", min_value=0, value=28, step=1)
    # w = st.sidebar.number_input(f"{group_label} - Crop width", min_value=10, value=1280, step=1)
    # h = st.sidebar.number_input(f"{group_label} - Crop height", min_value=10, value=984, step=1)
    crop_settings[group_label] = (x, y, w, h)

if st.button("▶️ Run Processing"):
    st.session_state['evaluation_triggered'] = True  # Mark evaluation started
    save_dir = create_save_directory()
    st.session_state['save_dir'] = save_dir

    # Process all uploaded groups
    if any([uploaded_files_group_1, uploaded_files_group_2, uploaded_files_group_3, uploaded_files_group_4]):
        st.write("### Processing Groups...")

        # Process and save data for each group
        if uploaded_files_group_1:
            process_and_save(uploaded_files_group_1, label_group_1, threshold, save_dir=save_dir)
        if uploaded_files_group_2:
            process_and_save(uploaded_files_group_2, label_group_2, threshold, save_dir=save_dir)
        if uploaded_files_group_3:
            process_and_save(uploaded_files_group_3, label_group_3, threshold, save_dir=save_dir)
        if uploaded_files_group_4:
            process_and_save(uploaded_files_group_4, label_group_4, threshold, save_dir=save_dir)

        # Generate statistics and visualizations
        st.write("### Combined Analysis and Visualizations")
        all_labels = [label_group_1, label_group_2, label_group_3, label_group_4]
        group_statistics = {}
        for label in all_labels:
            if areas_groups[label]:
                mean, median, std_dev = calculate_histogram_statistics(areas_groups[label])
                group_statistics[label] = {"Mean": mean, "Median": median, "Std Dev": std_dev}
                st.write(f"**{label}** - Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}")

        # Perform Mann-Whitney U Test and store results
        p_values = {}
        st.write("### Statistical Comparison (Mann-Whitney U Test)")

        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                label_1 = all_labels[i]
                label_2 = all_labels[j]

                # Ensure both groups have data
                if areas_groups[label_1] and areas_groups[label_2]:
                    stat, p_value = mannwhitneyu(areas_groups[label_1], areas_groups[label_2], alternative="two-sided")
                    p_values[f"{label_1} vs {label_2}"] = p_value
                    
                    # Display in Streamlit
                    st.write(f"**{label_1} vs {label_2}**: p-value = {p_value:.5f}")

        # Create layout with two columns for visualizations
        col1, col2 = st.columns(2)

        # --------- Histogram (Left Column) ---------
        with col1:
            fig_hist, ax_hist = plt.subplots(figsize=(10, 8))
            all_areas = [area for areas in areas_groups.values() for area in areas]
            bins = np.histogram_bin_edges(all_areas, bins=30)

            for label, areas in areas_groups.items():
                sns.histplot(areas, bins=bins, kde=False, label=label, ax=ax_hist, stat="percent", alpha=0.6)
                if label in group_statistics:
                    ax_hist.axvline(group_statistics[label]["Mean"], linestyle='--', label=f"{label} Mean")
            
            ax_hist.set_title("Histogram of Aggregation Areas", fontsize=22)
            ax_hist.set_xlabel("Aggregation Area (Pixels)", fontsize=20)
            ax_hist.set_ylabel("Percentage", fontsize=20)
            ax_hist.tick_params(axis='both', labelsize=20)  # Adjust tick label size
            ax_hist.legend()
            st.pyplot(fig_hist)

        # --------- Boxplot with Annotations (Right Column) ---------
        with col2:
            fig_box, ax_box = plt.subplots(figsize=(10, 8))
            filtered_log_transformed_areas = []
            valid_labels = []

            for i, label in enumerate(all_labels):
                if areas_groups[label]:
                    log_transformed = np.log1p(areas_groups[label])
                    filtered = remove_outliers(log_transformed)
                    filtered_log_transformed_areas.append(filtered)
                    valid_labels.append(label)

            sns.boxplot(data=filtered_log_transformed_areas, ax=ax_box, palette="Set2")
            sns.stripplot(data=filtered_log_transformed_areas, ax=ax_box, palette="dark", jitter=True)

            add_significance_annotations(ax_box, filtered_log_transformed_areas, valid_labels)

            ax_box.set_title("Box Plot of Log Aggregation Areas with Mann-Whitney U Test", fontsize=22)
            ax_box.set_ylabel("Log Aggregation Area", fontsize=20)
            ax_box.set_xticklabels(valid_labels, fontsize=20)
            ax_box.tick_params(axis='y', labelsize=20)
            st.pyplot(fig_box)

        # Store results in session state
        st.session_state['image_results'] = image_results
        st.session_state['areas_groups'] = areas_groups
        st.session_state['fig_hist'] = fig_hist
        st.session_state['fig_box'] = fig_box
        st.session_state['p_value'] = p_values


    else:
        st.warning("Please upload files in at least one group.")

# Reset evaluation flag after processing is done and after the download button is pressed.
# Generate and Download Results Section
if st.sidebar.button("Generate and Download Results"):
    if st.session_state.get('evaluation_triggered', False):
        st.session_state['download_triggered'] = True
        if st.session_state['image_results'] and st.session_state['save_dir']:
            with st.spinner("Preparing files for download..."):
                output_dir = os.path.join(st.session_state['save_dir'], "results")
                os.makedirs(output_dir, exist_ok=True)

                # Save results to CSV
                df = pd.DataFrame(st.session_state['image_results'])
                csv_path = os.path.join(output_dir, "summary_results.csv")
                df.to_csv(csv_path, index=False)
                # Save p-values to summary.txt inside the results folder
                summary_file = os.path.join(output_dir, "summary.txt")
                with open(summary_file, "w") as f:
                    f.write("### Statistical Comparison (Mann-Whitney U Test)\n")
                    for comparison, p_val in st.session_state['p_value'].items():
                        f.write(f"{comparison}: p-value = {p_val:.5f}\n")
                # Save visualizations as images
                hist_path = os.path.join(output_dir, "histogram.png")
                box_path = os.path.join(output_dir, "boxplot.png")
                st.session_state['fig_hist'].savefig(hist_path)
                st.session_state['fig_box'].savefig(box_path)

                # Create a ZIP file of the results
                zip_path = os.path.join(output_dir, "results.zip")
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(csv_path, arcname="summary_results.csv")
                    zipf.write(hist_path, arcname="histogram.png")
                    zipf.write(box_path, arcname="boxplot.png")
                    zipf.write(summary_file,arcname="summary.txt")

                # Provide download button
                with open(zip_path, "rb") as f:
                    st.download_button("Download All Results as ZIP", f, file_name="results.zip", mime="application/zip")

            # After download, reset flags for future processing
            st.session_state['download_triggered'] = False  # Reset download flag
            st.session_state['evaluation_triggered'] = False  # Reset evaluation flag to allow future processing
        else:
            st.warning("Please run the processing step first before downloading.")
    else:
        st.warning("Please run the processing step first before generating download.")



