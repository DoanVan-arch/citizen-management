import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Camera Test",
    page_icon="📷",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.main {padding: 20px;}
.stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Camera Test App")

# Initialize session state for camera
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

if 'camera' not in st.session_state:
    st.session_state.camera = None

# Function to toggle camera
def toggle_camera():
    if st.session_state.camera_on:
        # Turn off camera
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
        st.session_state.camera_on = False
    else:
        # Turn on camera
        st.session_state.camera = cv2.VideoCapture(0)
        if not st.session_state.camera.isOpened():
            st.error("Could not open camera!")
            st.session_state.camera = None
            return
        st.session_state.camera_on = True

# Create layout
col1, col2 = st.columns(2)

with col1:
    # Camera controls
    if st.session_state.camera_on:
        if st.button("Turn Off Camera"):
            toggle_camera()
    else:
        if st.button("Turn On Camera"):
            toggle_camera()

with col2:
    # Alternative: File upload
    st.subheader("Or upload an image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Create placeholder for camera feed
frame_placeholder = st.empty()

# Main camera loop
if st.session_state.camera_on and st.session_state.camera is not None:
    try:
        # Read frame from camera
        ret, frame = st.session_state.camera.read()
        if ret:
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
        else:
            st.error("Failed to capture frame from camera")
            toggle_camera()
    except Exception as e:
        st.error(f"Error accessing camera: {str(e)}")
        toggle_camera()

# Instructions
st.markdown("""---
### Instructions
1. Click 'Turn On Camera' to start your webcam
2. If the camera doesn't work, try uploading an image instead
3. Click 'Turn Off Camera' when you're done

### Troubleshooting
- Make sure your browser has permission to access the camera
- Try refreshing the page if the camera doesn't start
- Check if another application is using your camera
""")

# Clean up when the app is closed
def cleanup():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

# Register the cleanup function
st.experimental_singleton.clear()