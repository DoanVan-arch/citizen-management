import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

st.title("ZBar Test App")

st.write("This app tests if the ZBar library is properly installed.")

# Try to import and use pyzbar
try:
    # Create a simple QR code image for testing
    st.write("Creating a test image...")
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255  # Simple white square on black background
    
    st.image(img, caption="Test Image", width=200)
    
    # Try to decode (will not find anything but should not error)
    st.write("Testing pyzbar decode function...")
    results = decode(img)
    st.write(f"Decode results: {results}")
    
    st.success("✅ ZBar library is working correctly!")
    
    # Show camera input for QR scanning
    st.write("### Test with Camera")
    st.write("You can test scanning a QR code with your camera:")
    
    camera_image = st.camera_input("Scan QR Code")
    
    if camera_image is not None:
        # Process the captured image
        image = Image.open(camera_image)
        image_array = np.array(image)
        
        # Try to decode QR codes
        decoded_objects = decode(image_array)
        
        if decoded_objects:
            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8')
                st.success(f"Detected QR Code: {qr_data}")
                
                # Draw rectangle around the QR code
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    points = hull
                
                # Convert points to numpy array of shape (n,2)
                points = np.array([point for point in points], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Draw the polygon on the image
                cv2.polylines(image_array, [points], True, (0, 255, 0), 2)
                
                # Display the image with the QR code highlighted
                st.image(image_array, caption="Detected QR Code", use_column_width=True)
        else:
            st.warning("No QR code detected in the image.")
    
except Exception as e:
    st.error(f"Error with ZBar library: {str(e)}")
    st.error("Please check if libzbar is installed correctly.")
    
    # Provide installation instructions
    st.write("### Installation Instructions")
    st.code("""
    # For Ubuntu/Debian:
    sudo apt-get update
    sudo apt-get install -y libzbar0 libzbar-dev python3-zbar zbar-tools
    
    # Then reinstall pyzbar:
    pip install --force-reinstall pyzbar
    """, language="bash")