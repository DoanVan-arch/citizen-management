import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Simple Camera App",
    page_icon="ud83dudcf7",
    layout="wide"
)

# Title
st.title("Simple Camera App")

# Create tabs for different camera functions
tab1, tab2 = st.tabs(["Camera Feed", "Image Upload"])

with tab1:
    st.header("Camera Feed")
    
    # Use Streamlit's built-in camera input
    camera_image = st.camera_input("Take a picture with your camera")
    
    if camera_image is not None:
        # Display the captured image
        st.success("Image captured successfully!")
        
        # Process the image (convert to numpy array)
        image = Image.open(camera_image)
        image_array = np.array(image)
        
        # Example: Apply a simple image processing effect (grayscale)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_array, channels="RGB", use_container_width=True)
        
        with col2:
            st.subheader("Processed Image (Grayscale)")
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, use_container_width=True)

with tab2:
    st.header("Image Upload")
    
    # File uploader as alternative to camera
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.success("Image uploaded successfully!")
        
        # Process the image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Example: Apply a simple image processing effect (edge detection)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_array, channels="RGB", use_container_width=True)
        
        with col2:
            st.subheader("Processed Image (Edge Detection)")
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            st.image(edges, use_container_width=True)

# Instructions
st.markdown("""---
### Instructions
1. Use the "Camera Feed" tab to capture images with your camera
2. Use the "Image Upload" tab to upload existing images
3. The app will automatically process and display the images

### Troubleshooting
- Make sure your browser has permission to access the camera
- If the camera doesn't work, try using the image upload feature
- Some browsers may require HTTPS for camera access
""")