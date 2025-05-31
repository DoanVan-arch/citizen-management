import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Camera Test",
    page_icon="ud83dudcf7",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.main {padding: 20px;}
.stButton>button {width: 100%;}
.video-container {
    border: 2px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Camera Test App")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if 'frames' not in st.session_state:
    st.session_state.frames = []

# JavaScript for accessing camera
def get_camera_js():
    return """
    <div id="camera-container" class="video-container">
        <video id="camera" autoplay playsinline style="width: 100%; height: auto;"></video>
    </div>
    
    <script>
    // Function to access camera
    async function setupCamera() {
        const videoElement = document.getElementById('camera');
        const cameraContainer = document.getElementById('camera-container');
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false
            });
            
            videoElement.srcObject = stream;
            
            // Send frame to Streamlit every 100ms
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');
            
            function captureFrame() {
                if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
                    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    
                    // Send to Streamlit
                    window.parent.postMessage({type: 'camera_frame', frame: imageData}, '*');
                }
                
                // Continue capturing frames
                setTimeout(captureFrame, 100);
            }
            
            // Start capturing frames
            setTimeout(captureFrame, 500);
            
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            cameraContainer.innerHTML = `<div style="padding: 20px; color: red;">Error accessing camera: ${error.message}</div>`;
            return false;
        }
    }
    
    // Start camera when page loads
    document.addEventListener('DOMContentLoaded', () => {
        setupCamera().then(success => {
            if (success) {
                // Notify Streamlit that camera is active
                window.parent.postMessage({type: 'camera_status', status: 'active'}, '*');
            } else {
                window.parent.postMessage({type: 'camera_status', status: 'error'}, '*');
            }
        });
    });
    </script>
    """

# Create layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Camera Feed")
    
    # Display camera feed using JavaScript
    camera_placeholder = st.empty()
    camera_placeholder.markdown(get_camera_js(), unsafe_allow_html=True)
    
    # Display the most recent frame
    frame_display = st.empty()
    
    # Buttons for camera control
    col_a, col_b = st.columns(2)
    start_button = col_a.button("Start Camera")
    stop_button = col_b.button("Stop Camera")
    
    if start_button:
        st.session_state.camera_active = True
        st.success("Camera started! You should see the feed above.")
    
    if stop_button:
        st.session_state.camera_active = False
        st.info("Camera stopped.")
        frame_display.empty()

with col2:
    st.markdown("### Controls & Info")
    
    # Display camera status
    if st.session_state.camera_active:
        st.success("Camera is active")
    else:
        st.warning("Camera is inactive")
    
    # Capture button
    if st.button("Capture Frame") and st.session_state.camera_active:
        # In a real app, you would capture the current frame
        st.success("Frame captured!")
    
    # Alternative: File upload
    st.markdown("### Or upload an image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Instructions
st.markdown("""---
### Instructions
1. Allow browser access to your camera when prompted
2. The camera feed should appear in the top section
3. Use the buttons to control the camera
4. If the camera doesn't work, try uploading an image instead

### Troubleshooting
- Make sure your browser has permission to access the camera
- Try refreshing the page if the camera doesn't start
- Check if another application is using your camera
""")

# JavaScript to handle communication with Streamlit
st.markdown("""
<script>
// Listen for messages from the camera script
window.addEventListener('message', function(event) {
    if (event.data.type === 'camera_frame') {
        // Send the frame to Streamlit
        const data = {frame: event.data.frame};
        Streamlit.setComponentValue(data);
    }
    else if (event.data.type === 'camera_status') {
        // Update camera status
        const status = {status: event.data.status};
        Streamlit.setComponentValue(status);
    }
});
</script>
""", unsafe_allow_html=True)