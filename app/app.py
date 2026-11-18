
"""Hybrid SCD - Capstone Project Application.

Run:
    python -m streamlit run app.py

Local image processing and model inference.
"""

import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import content
from src.config import ALLOWED_TYPES, THRESHOLD, EFFNET_LAYER, GRADCAM_LAYER
from src.preprocessing import decode_image, preprocess_image, InvalidImageError


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid SCD | Capstone Project",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# COLOUR PALETTE
# =========================================================

MAROON = "#702D42"
MAROON_DARK = "#542031"
MAROON_PALE = "#F5EEF1"
WHITE = "#FFFFFF"
PAGE_GREY = "#F4F4F5"
BORDER = "#DCDDDF"
TEXT = "#292929"
MUTED = "#5F6368"


# =========================================================
# GLOBAL STYLING
# =========================================================

st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {MAROON};
        --background-color: {PAGE_GREY};
        --secondary-background-color: {WHITE};
        --text-color: {TEXT};
    }}

    /* Page surfaces */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background-color: {PAGE_GREY} !important;
        color: {TEXT} !important;
    }}

    .block-container {{
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Main content typography */
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] small,
    [data-testid="stMain"] strong,
    [data-testid="stMain"] em,
    [data-testid="stMain"] summary {{
        color: {TEXT} !important;
    }}

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {{
        color: {MAROON} !important;
        font-weight: 650 !important;
    }}

    [data-testid="stMain"] p,
    [data-testid="stMain"] li {{
        line-height: 1.65;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {WHITE} !important;
        border-right: 1px solid {BORDER} !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small {{
        color: {TEXT} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        background: {WHITE} !important;
        border-radius: 4px;
        padding: 8px 10px;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: {MAROON_PALE} !important;
        border-left: 3px solid {MAROON};
    }}

    /* Branding */
    .institution {{
        color: {MUTED} !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .brand {{
        color: {MAROON} !important;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 8px;
    }}

    .page-subtitle {{
        color: {MUTED} !important;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }}

    .page-rule {{
        width: 52px;
        height: 3px;
        background: {MAROON};
        margin: 9px 0 17px;
    }}

    /* Research header and panels */
    .research-header,
    .content-panel {{
        background: {WHITE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-left: 4px solid {MAROON} !important;
        border-radius: 4px;
        padding: 22px 25px;
        margin-bottom: 18px;
        line-height: 1.7;
    }}

    .research-header h1 {{
        color: {MAROON} !important;
    }}

    .research-header p,
    .content-panel p {{
        color: {TEXT} !important;
    }}

    /* Notices */
    .notice {{
        background: {MAROON_PALE} !important;
        color: {TEXT} !important;
        border: 1px solid #E4D2D9 !important;
        border-left: 4px solid {MAROON} !important;
        border-radius: 4px;
        padding: 13px 16px;
        margin: 12px 0 20px;
        line-height: 1.65;
    }}

    .notice strong,
    .notice b {{
        color: {MAROON} !important;
    }}

    /* Alerts: info, success, warning and error */
    [data-testid="stAlert"] {{
        background: {MAROON_PALE} !important;
        border: 1px solid #E4D2D9 !important;
        border-left: 4px solid {MAROON} !important;
        color: {TEXT} !important;
    }}

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {{
        color: {TEXT} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {{
        background: {WHITE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 5px;
    }}

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: {MAROON} !important;
        color: {WHITE} !important;
        border: 1px solid {MAROON} !important;
    }}

    [data-testid="stFileUploader"] button p,
    [data-testid="stFileUploader"] button span {{
        color: {WHITE} !important;
    }}

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] input {{
        background: {WHITE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}

    /* Buttons: dark maroon surface, white text */
    div.stButton > button,
    div.stDownloadButton > button {{
        background: {MAROON} !important;
        color: {WHITE} !important;
        border: 1px solid {MAROON} !important;
        border-radius: 4px;
        font-weight: 600;
    }}

    div.stButton > button p,
    div.stButton > button span,
    div.stDownloadButton > button p,
    div.stDownloadButton > button span {{
        color: {WHITE} !important;
    }}

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {{
        background: {MAROON_DARK} !important;
        border-color: {MAROON_DARK} !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: {WHITE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px;
        padding: 14px;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {{
        color: {MUTED} !important;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {{
        color: {MAROON} !important;
    }}

    /* Tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background: {WHITE} !important;
        border: 1px solid {BORDER} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {WHITE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: {TEXT} !important;
    }}

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div {{
        background: {MAROON} !important;
    }}

    /* Architecture diagram */
    .flow-box {{
        display: inline-block;
        background: {WHITE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 10px 13px;
        margin: 4px;
        vertical-align: middle;
        line-height: 1.5;
    }}

    .flow-box b {{
        color: {MAROON} !important;
    }}

    .flow-branch {{
        background: {MAROON_PALE} !important;
        border-color: #D9C2CA;
    }}

    .flow-output {{
        border: 1px solid {MAROON};
        font-weight: 700;
    }}

    .flow-arrow {{
        color: {MAROON} !important;
        font-weight: 700;
    }}

    /* Footer */
    .footer {{
        border-top: 1px solid {BORDER};
        padding-top: 14px;
        margin-top: 32px;
        text-align: center;
        color: {MUTED} !important;
        font-size: 0.75rem;
    }}

    hr {{
        border-color: {BORDER} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

SECTIONS = [
    "Research Overview",
    "Blood-Smear Image Analysis",
    "Prediction and Grad-CAM Explainability",
    "Experimental Results",
    "Model Architecture",
]

with st.sidebar:

    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwtEU6hkp9wy6Fc_-4zbaAzAHbOvgPPmtCYicnD0hSGQ&s=10",
        width=245,
    )

    st.markdown(
        f"""
        <div style="padding:8px 2px 17px 2px;">
            <div class="brand">HYBRID SCD</div>
            <div style="font-size:0.82rem;color:{MUTED};">
                Capstone Project
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        f"""
        <div style="
            font-size:0.72rem;
            color:{MUTED};
            font-weight:700;
            letter-spacing:0.8px;
            margin-bottom:8px;">
            PROJECT SECTIONS
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        SECTIONS,
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.divider()
    st.caption("Capstone Project")
    st.caption("Academic research prototype")


# =========================================================
# SHARED FUNCTIONS
# =========================================================

def page_heading(title, subtitle):
    st.markdown(
        f"""
        <div class="institution">CAPSTONE PROJECT</div>
        <h1 style="color:{MAROON} !important;">{title}</h1>
        <div class="page-rule"></div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def get_model():
    from src.model_loader import load_model, ModelLoadError

    try:
        return load_model(), None
    except ModelLoadError as exc:
        return None, str(exc)


# Research disclaimer
st.markdown(
    f"""
    <div class="notice">
        <b>Research disclaimer:</b> {content.DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1. RESEARCH OVERVIEW
# =========================================================

if page == SECTIONS[0]:

    st.markdown(
        f"""
        <div class="research-header">
            <div class="institution">CAPSTONE PROJECT</div>
            <h1>{content.TITLE}</h1>
            <p>{content.SUBTITLE}</p>
            <p style="margin-top:8px;font-size:0.85rem;">
                {content.AFFILIATION}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_heading(
        "Project Overview",
        "Background, contributions, dataset and study limitations.",
    )

    st.subheader("Background")

    st.markdown(
        f'<div class="content-panel">{content.BACKGROUND}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Project Contributions")

    for index, item in enumerate(content.CONTRIBUTIONS, start=1):
        st.markdown(f"**{index}.** {item}")

    st.divider()

    st.subheader("Dataset")

    st.write(
        "The collection contains 1,052 blood-smear images "
        "(184 normal and 868 sickle; approximately 83% positive)."
    )

    st.markdown(
        """
        <div class="notice">
            The dataset name, source and citation have not yet been
            specified in the manuscript draft.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Study Limitations")

    for item in content.LIMITATIONS:
        st.markdown(f"- {item}")


# =========================================================
# 2. BLOOD-SMEAR IMAGE ANALYSIS
# =========================================================

elif page == SECTIONS[1]:

    page_heading(
        "Blood-Smear Image Analysis",
        "Upload and preprocess a blood-smear image using the implemented pipeline.",
    )

    st.subheader("Image Preprocessing")

    st.write(
        "The preprocessing pipeline applies BGR-to-RGB conversion, "
        "resizing to 224 × 224 using INTER_AREA interpolation, "
        "3 × 3 median filtering, Y-channel histogram equalization "
        "in YCrCb colour space, and conversion back to RGB."
    )

    st.divider()

    uploaded = st.file_uploader(
        "`Upload blood-smear image",
        type=ALLOWED_TYPES,
    )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        if st.session_state.get("image_hash") != image_hash:

            try:
                bgr = decode_image(image_bytes)
                rgb_pre = preprocess_image(bgr)

                st.session_state["image_hash"] = image_hash
                st.session_state["file_name"] = uploaded.name
                st.session_state["original_rgb"] = bgr[:, :, ::-1].copy()
                st.session_state["preprocessed"] = rgb_pre

                st.session_state.pop("result", None)

            except InvalidImageError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(f"Image processing failed: {exc}")

    if "preprocessed" in st.session_state:

        st.markdown(
            f"**Loaded image:** {st.session_state['file_name']}"
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Original Image")
            st.image(
                st.session_state["original_rgb"],
                caption="Uploaded blood-smear image",
                width="stretch",
            )

        with col2:
            st.subheader("Preprocessed Image")
            st.image(
                st.session_state["preprocessed"],
                caption="224 × 224 RGB image",
                width="stretch",
            )

        st.markdown(
            """
            <div class="notice">
                Image preprocessing completed. Continue to the
                prediction section.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 3. PREDICTION AND GRAD-CAM
# =========================================================

elif page == SECTIONS[2]:

    page_heading(
        "Prediction and Grad-CAM Explainability",
        "Single-image inference and EfficientNet branch activation visualization.",
    )

    if "preprocessed" not in st.session_state:

        st.warning(
            "No image has been uploaded. Please use Blood-Smear "
            "Image Analysis first."
        )

    else:

        st.markdown(
            f"**Current image:** {st.session_state['file_name']}"
        )

        model, error = get_model()

        if error:
            st.error(error)

        elif st.button("Run Prediction and Grad-CAM", type="primary"):

            with st.spinner("Running model inference..."):

                try:
                    from src.inference import predict
                    from src.gradcam import compute_gradcam, overlay

                    rgb = st.session_state["preprocessed"]

                    prediction = predict(model, rgb)
                    heatmap = compute_gradcam(model, rgb)
                    blended = overlay(rgb, heatmap)

                    st.session_state["result"] = (
                        prediction,
                        heatmap,
                        blended,
                    )

                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

        if "result" in st.session_state:

            prediction, heatmap, blended = st.session_state["result"]

            st.subheader("Model Output")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "P(sickle class)",
                    f"{prediction.probability:.4f}",
                )

            with col2:
                st.metric(
                    f"Predicted class at threshold {THRESHOLD}",
                    prediction.label,
                )

            with col3:
                st.metric(
                    "Margin from threshold",
                    f"{prediction.probability - THRESHOLD:+.4f}",
                )

            st.progress(
                min(max(prediction.probability, 0.0), 1.0)
            )

            st.caption(
                "This is the output of the research model for one "
                "uploaded image. It is not a clinical diagnosis."
            )

            st.divider()

            st.subheader("Grad-CAM Activation Visualization")

            st.write(
                f"The visualization is computed from the nested "
                f"{EFFNET_LAYER} branch at {GRADCAM_LAYER}. "
                "The gradient target is the mean of the EfficientNet "
                "branch output (a proxy loss), not the final sigmoid "
                "class score. Therefore, this is a branch activation "
                "visualization, not a class-specific explanation of "
                "the final hybrid prediction. It does not represent "
                "the residual CNN branch."
            )

            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                st.subheader("Input Image")
                st.image(
                    st.session_state["preprocessed"],
                    width="stretch",
                )

            with col2:
                st.subheader("Activation Map")

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(heatmap, cmap="Reds", vmin=0, vmax=1)
                ax.axis("off")
                fig.tight_layout(pad=0)

                buffer = io.BytesIO()

                fig.savefig(
                    buffer,
                    format="png",
                    bbox_inches="tight",
                    facecolor="white",
                )

                plt.close(fig)

                st.image(
                    buffer.getvalue(),
                    width="stretch",
                )

            with col3:
                st.subheader("Overlay")
                st.image(
                    blended,
                    width="stretch",
                )


# =========================================================
# 4. EXPERIMENTAL RESULTS
# =========================================================

elif page == SECTIONS[3]:

    page_heading(
        "Experimental Results",
        "Reported evaluation results for the proposed hybrid model.",
    )

    st.write(
        "The proposed hybrid model was evaluated on a held-out test "
        "set of 156 images: 26 normal and 130 sickle. The decision "
        "threshold was 0.5."
    )

    metrics = list(content.HYBRID_METRICS.items())
    metric_columns = st.columns(4)

    for index, (key, value) in enumerate(metrics):
        with metric_columns[index % 4]:
            st.metric(key, value)

    with st.expander("Full Reported Metrics and Confidence Intervals"):

        st.dataframe(
            pd.DataFrame(
                metrics,
                columns=["Metric", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.warning(
        f"Always predicting sickle scores "
        f"{content.ALWAYS_POSITIVE_BASELINE} on this test set, "
        f"and the accuracy interval contains that baseline. "
        f"Normal class: {content.NORMAL_CLASS}"
    )

    st.divider()

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:

        st.subheader("Confusion Matrix")

        cm = content.HYBRID_CM

        matrix = np.array([
            [cm["TN"], cm["FP"]],
            [cm["FN"], cm["TP"]],
        ])

        fig, ax = plt.subplots(figsize=(4.2, 3.7))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)

        ax.imshow(
            matrix,
            cmap="Reds",
            interpolation="nearest",
        )

        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=TEXT,
                fontsize=16,
                fontweight="bold",
            )

        ax.set_xticks(
            [0, 1],
            ["Predicted normal", "Predicted sickle"],
        )

        ax.set_yticks(
            [0, 1],
            ["True normal", "True sickle"],
        )

        ax.set_title(
            "Held-out Test Set",
            color=TEXT,
            pad=12,
        )

        ax.tick_params(colors=TEXT, length=0)

        for spine in ax.spines.values():
            spine.set_color(BORDER)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "TN: True Negative | FP: False Positive | "
            "FN: False Negative | TP: True Positive"
        )

    with col2:

        st.subheader("Accuracy by Source Folder")

        st.dataframe(
            pd.DataFrame(
                content.SUBGROUP,
                columns=[
                    "Test images from",
                    "Count",
                    "Correct",
                    "Accuracy",
                ],
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Annotated positives were detected more often than "
            "unannotated ones, consistent with a probable "
            "annotation shortcut."
        )

    st.divider()

    st.subheader("Comparison with Other Models in the Study")

    comparison = pd.DataFrame(
        content.COMPARISON,
        columns=content.COMPARISON_COLUMNS,
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(content.COMPARISON_NOTE)


# =========================================================
# 5. MODEL ARCHITECTURE
# =========================================================

elif page == SECTIONS[4]:

    page_heading(
        "Model Architecture",
        "Dual-branch feature extraction and classification architecture.",
    )

    st.write(content.ARCHITECTURE_TEXT)

    st.divider()

    st.subheader("Architecture Flow")

    st.markdown(
        """
        <div>
            <span class="flow-box">
                Input<br>224 × 224 × 3
            </span>
            <span class="flow-arrow">→</span>
            <span class="flow-box">
                Augmentation<br>Training only
            </span>
        </div>

        <div style="margin-top:16px;">
            <span class="flow-box flow-branch">
                <b>Branch A: EfficientNet-B0</b><br>
                ImageNet pretrained<br>
                GAP → Dropout<br>
                1280 features
            </span>
        </div>

        <div style="margin-top:10px;">
            <span class="flow-box flow-branch">
                <b>Branch B: Residual CNN</b><br>
                Residual blocks and SE attention<br>
                GAP → Dense 256 → Dropout
            </span>
        </div>

        <div style="margin-top:10px;">
            <span class="flow-arrow">↓</span>
            <span class="flow-box">
                Concatenation · 1536 features
            </span>
            <span class="flow-arrow">→</span>
            <span class="flow-box">Batch Normalization</span>
            <span class="flow-arrow">→</span>
            <span class="flow-box">Dense 256</span>
            <span class="flow-arrow">→</span>
            <span class="flow-box">Dropout</span>
            <span class="flow-arrow">→</span>
            <span class="flow-box">Dense 64</span>
            <span class="flow-arrow">→</span>
            <span class="flow-box flow-output">
                Sigmoid<br>P(sickle class)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Training Configuration")

    for item in content.TRAINING_TEXT:
        st.markdown(f"- {item}")

    st.divider()

    st.subheader("Loaded Model Information")

    model, error = get_model()

    if error:

        st.error(error)

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Name", model.name)

        with col2:
            st.metric("Parameters", f"{model.count_params():,}")

        with col3:
            st.metric(
                "Input → Output",
                f"{model.input_shape[1:]} → {model.output_shape[1:]}",
            )

        rows = []

        for layer in model.layers:

            try:
                output_shape = str(layer.output.shape)
            except Exception:
                output_shape = ""

            rows.append(
                (
                    layer.name,
                    layer.__class__.__name__,
                    output_shape,
                )
            )

        with st.expander(
            f"Top-level Model Layers ({len(rows)})"
        ):

            st.dataframe(
                pd.DataFrame(
                    rows,
                    columns=[
                        "Layer",
                        "Type",
                        "Output Shape",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        HYBRID SCD | CAPSTONE PROJECT<br>
        Academic research prototype | Non-diagnostic
    </div>
    """,
    unsafe_allow_html=True,
)

