import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pyzbar.pyzbar import decode

# Set page config
st.set_page_config(
    page_title="Camera App",
    page_icon="ud83dudcf7",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.main {padding: 20px;}
.stButton>button {width: 100%;}
.camera-feed {border: 2px solid #ddd; border-radius: 10px; overflow: hidden;}
.info-card {padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Camera Application")

# Initialize session state
if 'qr_data' not in st.session_state:
    st.session_state.qr_data = None

if 'last_image' not in st.session_state:
    st.session_state.last_image = None

# Function to process image for QR code detection
def process_image_for_qr(image):
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale for QR detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect QR codes
    decoded_objects = decode(gray)
    
    # Draw bounding box around QR codes
    for obj in decoded_objects:
        # Extract polygon points
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            cv2.polylines(image, [hull], True, (0, 255, 0), 2)
        else:
            # Draw rectangle
            cv2.polylines(image, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        
        # Get data
        qr_data = obj.data.decode('utf-8')
        st.session_state.qr_data = qr_data
        
        # Display data on image
        cv2.putText(image, "QR: " + qr_data[:20] + "...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return image

# Function to process uploaded image
def process_uploaded_image(uploaded_file):
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Process for QR code
    processed_image = process_image_for_qr(image)
    
    # Save processed image
    st.session_state.last_image = processed_image
    
    return processed_image

# Create tabs for different camera functions
tab1, tab2 = st.tabs(["Camera Feed", "QR Code Scanner"])

with tab1:
    st.markdown("""
    <div class="info-card">
    <h3>Camera Feed</h3>
    <p>Use your camera to capture images or video</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera input
    camera_image = st.camera_input("Take a picture", key="camera_input1")
    
    if camera_image is not None:
        # Process the captured image
        image = Image.open(camera_image)
        image_array = np.array(image)
        
        # Display the processed image
        st.image(image_array, caption="Captured Image", use_container_width=True)
        
        # Save button
        if st.button("Save Image", key="save_btn1"):
            # Create temp directory if it doesn't exist
            if not os.path.exists("temp_images"):
                os.makedirs("temp_images")
            
            # Save image
            timestamp = np.datetime64('now')
            filename = f"temp_images/captured_image_{timestamp}.jpg".replace(":", "-")
            Image.fromarray(image_array).save(filename)
            st.success(f"Image saved as {filename}")

with tab2:
    st.markdown("""
    <div class="info-card">
    <h3>QR Code Scanner</h3>
    <p>Scan QR codes using your camera</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera input for QR scanning
    qr_camera_image = st.camera_input("Scan QR Code", key="camera_input2")
    
    # File uploader as alternative
    st.markdown("### Or upload an image with QR code")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process camera image for QR
    if qr_camera_image is not None:
        # Process the captured image
        image = Image.open(qr_camera_image)
        image_array = np.array(image)
        
        # Process for QR code
        processed_image = process_image_for_qr(image_array)
        
        # Display the processed image
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        # Display QR data if detected
        if st.session_state.qr_data:
            st.success(f"QR Code detected: {st.session_state.qr_data}")
            
            # Parse QR data (assuming it's in JSON or key-value format)
            try:
                # Display in a more structured way
                st.json(st.session_state.qr_data)
            except:
                # If not JSON, just display as text
                st.text(st.session_state.qr_data)
    
    # Process uploaded file for QR
    elif uploaded_file is not None:
        processed_image = process_uploaded_image(uploaded_file)
        st.image(processed_image, caption="Processed Uploaded Image", use_container_width=True)
        
        # Display QR data if detected
        if st.session_state.qr_data:
            st.success(f"QR Code detected: {st.session_state.qr_data}")
            
            # Parse QR data
            try:
                st.json(st.session_state.qr_data)
            except:
                st.text(st.session_state.qr_data)

# Instructions
st.markdown("""---
### Instructions
1. Select the tab for the function you want to use
2. Allow camera access when prompted
3. For QR scanning, point your camera at a QR code
4. You can also upload images with QR codes

### Troubleshooting
- Make sure your browser has permission to access the camera
- Try refreshing the page if the camera doesn't start
- If QR codes aren't detected, make sure they're clearly visible and well-lit
""")