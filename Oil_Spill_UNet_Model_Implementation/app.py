"""
Streamlit Web Application for Oil Spill Detection
Clean and modern UI for real-time oil spill detection
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import os

from model import AttentionUNet

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding-top: 2rem;
    }

    /* Header styling */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(120deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Upload section */
    .uploadedFile {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8FAFC;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model (cached for efficiency)"""
    try:
        # Construct absolute path to model file
        model_path = os.path.join(SCRIPT_DIR, 'best_model.pth')

        model = AttentionUNet(in_channels=1, out_channels=1)
        model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.error(f"Looking for model at: {os.path.join(SCRIPT_DIR, 'best_model.pth')}")
        st.error("Make sure 'best_model.pth' exists in the project directory.")
        st.error("Run 'python train.py' first to train the model.")
        return None


def preprocess_image(image):
    """
    Preprocess uploaded image following training pipeline.

    Args:
        image: PIL Image

    Returns:
        img_tensor: Preprocessed tensor [1, 1, 256, 256]
        img_display: Image for display (256x256 numpy array)
    """
    # Convert to grayscale
    image = image.convert('L')

    # Resize to 256x256
    image = image.resize((256, 256))

    # Convert to numpy
    img_np = np.array(image)

    # Median blur (following training preprocessing)
    img_np = cv2.medianBlur(img_np, 3)

    # Normalize to [0, 1]
    img_normalized = img_np.astype(np.float32) / 255.0

    # Create tensor [1, 1, 256, 256]
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

    return img_tensor, img_np


def calculate_area(mask, pixel_resolution=10):
    """
    Calculate oil spill area in km².

    Args:
        mask: Binary mask (0/1)
        pixel_resolution: Resolution in meters per pixel (default: 10m for Sentinel-1)

    Returns:
        Area in km²
    """
    oil_pixels = mask.sum()
    area_m2 = oil_pixels * (pixel_resolution ** 2)
    area_km2 = area_m2 / 1e6
    return area_km2


def main():
    # Header
    st.markdown(
        '<div class="main-header">🌊 Oil Spill Detection System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">AI-powered detection from satellite imagery using Attention U-Net</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("---")

        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Confidence threshold for oil spill detection (higher = more strict)"
        )

        pixel_resolution = st.number_input(
            "Pixel Resolution (meters)",
            min_value=1,
            max_value=100,
            value=10,
            help="Spatial resolution of satellite image (meters per pixel). Default: 10m for Sentinel-1 SAR"
        )

        st.markdown("---")
        st.markdown("### 📊 About This Model")
        st.markdown("""
        **Architecture:** Attention U-Net

        **Features:**
        - Attention mechanisms for better feature focusing
        - Trained on 8,070 satellite images
        - Binary segmentation (oil vs background)
        - Real-time inference on CPU/GPU

        **Performance:**
        - Dice Score: 0.78
        - IoU: 0.67
        - Precision: 0.80
        - Recall: 0.84
        - High accuracy and reliability

        **Model Details:**
        - Parameters: 31.4M
        - Training Time: ~30 minutes
        - Inference Time: <1 second per image
        """)

        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload Image**: Click the upload button and select a satellite image (SAR preferred)
        2. **Adjust Settings**: Modify detection threshold and pixel resolution if needed
        3. **Analyze**: Click "Analyze Image" button to run detection
        4. **Review Results**: View the detection overlay, metrics, and probability heatmap
        5. **Download**: Save the binary mask, overlay image, or probability map
        6. **Interpret**: Green alert means no spill, red alert means oil spill detected

        **Tips:**
        - Use SAR (Synthetic Aperture Radar) images for best results
        - Higher threshold = stricter detection (fewer false positives)
        - Lower threshold = more sensitive (catch small spills)
        - Default settings work well for most cases
        """)

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Main content
    st.markdown("### 📤 Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Choose a satellite image (SAR or optical imagery)",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload satellite imagery (SAR recommended for best results)"
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Create columns for side-by-side display
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("#### 📷 Input Image")
            st.image(image, use_container_width=True)

        # Analyze button
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("🔄 Analyzing image... Please wait."):
                # Preprocess
                img_tensor, img_display = preprocess_image(image)

                # Predict
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.sigmoid(output).squeeze().numpy()
                    mask = (prob > threshold).astype(np.uint8)

                # Calculate metrics
                oil_pixels = int(mask.sum())
                area_km2 = calculate_area(mask, pixel_resolution)
                coverage = (oil_pixels / (256 * 256)) * 100
                max_confidence = float(prob.max())

                # Display result - Detection Overlay
                with col2:
                    st.markdown("#### 🎯 Detection Overlay")

                    # Create overlay visualization
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img_display, cmap='gray')
                    ax.imshow(mask, cmap='Reds', alpha=0.6 * (mask > 0))
                    ax.axis('off')
                    ax.set_title('Oil Spill Detection', fontsize=14, fontweight='bold', pad=15)
                    st.pyplot(fig)
                    plt.close()

                # Display Binary Mask
                with col3:
                    st.markdown("#### 🎭 Binary Mask")

                    # Create binary mask visualization
                    fig3, ax3 = plt.subplots(figsize=(6, 6))
                    ax3.imshow(mask, cmap='binary_r')  # White = oil, Black = water
                    ax3.axis('off')
                    ax3.set_title('Segmentation Mask', fontsize=14, fontweight='bold', pad=15)
                    st.pyplot(fig3)
                    plt.close()

                    # Add legend
                    st.caption("⬜ White = Oil Spill | ⬛ Black = Water")

                # Results section
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")

                # Key metrics - 4 columns
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    st.metric(
                        label="🔢 Oil Pixels",
                        value=f"{oil_pixels:,}",
                        delta="Detected" if oil_pixels > 50 else "Minimal",
                        help="Number of pixels classified as oil spill"
                    )

                with metric_cols[1]:
                    st.metric(
                        label="📏 Estimated Area",
                        value=f"{area_km2:.3f} km²",
                        delta=f"{area_km2*1000000:.0f} m²",
                        help=f"Area calculation based on {pixel_resolution}m resolution"
                    )

                with metric_cols[2]:
                    st.metric(
                        label="📊 Coverage",
                        value=f"{coverage:.2f}%",
                        delta="High" if coverage > 10 else "Low",
                        help="Percentage of image covered by oil spill"
                    )

                with metric_cols[3]:
                    st.metric(
                        label="🎯 Max Confidence",
                        value=f"{max_confidence:.1%}",
                        help="Highest confidence score in the prediction"
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Status alert
                if oil_pixels > 50:
                    st.error(
                        f"🚨 **Oil Spill Detected!**\n\n"
                        f"Estimated area: **{area_km2:.3f} km²** ({oil_pixels:,} pixels)\n\n"
                        f"Coverage: **{coverage:.2f}%** of image area\n\n"
                        f"Immediate action recommended for environmental protection."
                    )
                else:
                    st.success(
                        "✅ **No Significant Oil Spill Detected**\n\n"
                        "The analyzed area appears clean. Continue monitoring as needed."
                    )

                # Probability Heatmap
                st.markdown("---")
                st.markdown("### 🔥 Probability Heatmap")
                st.markdown("Shows the model's confidence for each pixel (red = high probability of oil spill)")

                fig2, ax2 = plt.subplots(figsize=(10, 8))
                im = ax2.imshow(prob, cmap='YlOrRd', vmin=0, vmax=1)
                ax2.axis('off')
                ax2.set_title(
                    'Oil Spill Probability Map',
                    fontsize=16,
                    fontweight='bold',
                    pad=20
                )
                cbar = plt.colorbar(im, ax=ax2, fraction=0.046, label='Probability')
                cbar.set_label('Probability', fontsize=12)
                st.pyplot(fig2)
                plt.close()

                st.info("""
                **How to interpret the heatmap:**
                - 🟥 **Red areas**: High probability of oil spill (>70%)
                - 🟧 **Orange areas**: Medium probability (40-70%)
                - 🟨 **Yellow areas**: Low probability (10-40%)
                - ⬜ **Light areas**: Very low probability (<10%)
                """)

                # Download section
                st.markdown("---")
                st.markdown("### 💾 Download Results")

                download_cols = st.columns(3)

                with download_cols[0]:
                    # Binary mask
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    mask_img.save(buf, format='PNG')
                    buf.seek(0)

                    st.download_button(
                        label="📥 Download Binary Mask",
                        data=buf,
                        file_name=f"mask_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_cols[1]:
                    # Overlay image
                    fig_download, ax_download = plt.subplots(figsize=(8, 8))
                    ax_download.imshow(img_display, cmap='gray')
                    ax_download.imshow(mask, cmap='Reds', alpha=0.6 * (mask > 0))
                    ax_download.axis('off')

                    buf2 = io.BytesIO()
                    fig_download.savefig(buf2, format='PNG', bbox_inches='tight', dpi=150)
                    buf2.seek(0)
                    plt.close(fig_download)

                    st.download_button(
                        label="📥 Download Overlay Image",
                        data=buf2,
                        file_name=f"detection_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_cols[2]:
                    # Probability map
                    prob_img = Image.fromarray((prob * 255).astype(np.uint8))
                    buf3 = io.BytesIO()
                    prob_img.save(buf3, format='PNG')
                    buf3.seek(0)

                    st.download_button(
                        label="📥 Download Probability Map",
                        data=buf3,
                        file_name=f"probability_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True
                    )

    else:
        # Welcome message
        st.info("👆 **Upload a satellite image to get started**")

        # Sample info in expander
        with st.expander("ℹ️ What kind of images work best?"):
            st.markdown("""
            **Best Image Types:**
            - SAR (Synthetic Aperture Radar) satellite images
            - Grayscale satellite imagery
            - Sentinel-1 imagery (10m resolution)
            - Size: Any size (will be automatically resized to 256×256)

            **Image Sources:**
            - Sentinel-1 from ESA Copernicus Open Access Hub
            - Kaggle oil spill datasets
            - NASA Earthdata
            - Other SAR satellite providers

            **Supported Formats:**
            - PNG, JPG, JPEG, TIF, TIFF
            - Both grayscale and RGB (will be converted to grayscale)

            **Note:** The model was trained on SAR imagery and works best with similar data.
            Optical imagery may produce less accurate results.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>"
        "AI Powered Oil Spill Detection System"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
