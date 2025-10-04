import streamlit as st
from PIL import Image
import requests
import io

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Pest Detection & Analysis",
    page_icon="🐛",
    layout="wide",
)

# --- 2. Backend API Configuration ---
# This is the address of your locally running FastAPI server
API_BASE_URL = "http://127.0.0.1:8000"

# --- 3. Helper Functions to Call Backend ---
def predict_pest(image_bytes):
    """Sends image to the /predict endpoint and returns the result."""
    try:
        files = {'file': ('pest_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Analysis Failed: Could not connect to the backend. Please ensure the API server is running. Error: {e}")
        return None

# --- 4. Session State Management ---
# We use session_state to control what is shown on the page.
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# --- 5. Page Rendering Logic ---

if st.session_state.page == 'upload':
    # --- UI for Upload Page ---
    st.markdown("""
        <div style="background-color: #22c55e; color: white; padding: 2rem 1rem; text-align: center; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: bold;">AI-Powered Pest Detection</h1>
            <p style="font-size: 1.1rem; max-width: 700px; margin: auto; opacity: 0.9;">
                Upload an image of any pest affecting your crops and get instant identification plus personalized treatment recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.header("Upload Your Pest Image Here")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Your Uploaded Image', use_column_width=True)
        
        # When the button is clicked, we process the image
        if st.button('Analyze Pest', use_container_width=True, type="primary"):
            with st.spinner("🔍 Analyzing image... Please wait."):
                image_bytes = uploaded_file.getvalue()
                prediction_result = predict_pest(image_bytes)
                
                # If prediction is successful, store results and switch page state
                if prediction_result and prediction_result.get("predictions"):
                    st.session_state.prediction_result = prediction_result
                    st.session_state.uploaded_image = image_bytes
                    st.session_state.page = 'results'
                    st.rerun() # Rerun the script to show the results page

    # --- How It Helps Section ---
    st.markdown("---")
    st.header("How It Helps")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("Accurate Identification")
        st.write("Our advanced AI model is trained on thousands of pest images to provide precise identification.")
    with col2:
        st.success("Actionable Recommendations")
        st.write("Receive clear, expert-backed advice on both organic (IPM) and chemical solutions.")
    with col3:
        st.success("Prevention Strategies")
        st.write("Learn how to protect your crops in the future with our detailed prevention tips.")


elif st.session_state.page == 'results':
    # --- UI for Results Page ---
    st.title("Pest Analysis Results")
    
    # Button to go back to the upload page
    if st.button("⬅️ Analyze Another Image"):
        st.session_state.page = 'upload'
        st.session_state.prediction_result = None
        st.session_state.uploaded_image = None
        st.rerun()

    prediction = st.session_state.prediction_result['predictions'][0]
    
    with st.container(border=True):
        st.header("Pest Identification")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state.uploaded_image, caption="Your Uploaded Image", use_column_width=True)
        with col2:
            pest_display_name = prediction['class_name'].replace('_', ' ').title()
            st.markdown(f"## {pest_display_name}")
            
            confidence_pct = prediction['confidence'] * 100
            
            if confidence_pct >= 80:
                badge_text = "High Confidence"
            elif confidence_pct >= 60:
                badge_text = "Medium Confidence"
            else:
                badge_text = "Low Confidence"
                
            st.markdown(f"**Confidence Score:** `{confidence_pct:.2f}%` ({badge_text})")

    # --- Placeholder for Knowledge Base Info ---
    st.markdown("---")
    st.header("Detailed Information & Recommendations")
    st.info("This section will be populated with detailed pest information, treatment plans, and prevention tips from our knowledge base in a future update.")

