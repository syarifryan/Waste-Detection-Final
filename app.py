from pathlib import Path
import streamlit as st
import helper
import settings
import os

st.set_page_config(
    page_title="Waste Detection",
    layout="wide"
)

st.sidebar.title("Detect Console")

model_path = Path(settings.DETECTION_MODEL)

st.title("Intelligent Waste Detection System")
st.write("""
This system helps classify waste into recyclable, non-recyclable, and hazardous categories. 
You can either use real-time webcam detection or upload images/videos for analysis.
""")

# Create results directory if it doesn't exist
RESULTS_DIR = "detection_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

st.markdown(
"""
<style>
    .stRecyclable {
        background-color: rgba(233,192,78,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    .stNonRecyclable {
        background-color: rgba(94,128,173,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    .stHazardous {
        background-color: rgba(194,84,85,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }
    [data-testid="stExpander"] .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""",
unsafe_allow_html=True
)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Detection mode selection
detection_mode = st.radio(
    "Select Detection Mode:",
    ("Webcam", "Upload Image", "Upload Video"),
    horizontal=True
)

if detection_mode == "Webcam":
    helper.play_webcam(model)
elif detection_mode == "Upload Image":
    helper.upload_image(model, RESULTS_DIR)
elif detection_mode == "Upload Video":
    helper.upload_video(model, RESULTS_DIR)

# Display previous results section
with st.expander("View Previous Detection Results"):
    helper.display_previous_results(RESULTS_DIR)

st.sidebar.markdown("""
### About
This system uses YOLO object detection to classify waste into:
- Recyclable ♻️
- Non-Recyclable 🗑️
- Hazardous ☣️

Detection results are saved locally for future reference.
""", unsafe_allow_html=True)