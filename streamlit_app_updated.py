from operator import truediv
import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import av
from contextlib import contextmanager

# Thu00eam try-except cho import asyncio u0111u1ec3 xu1eed lu00fd lu1ed7i liu00ean quan u0111u1ebfn asyncio
try:
    import asyncio
    import threading
    import uuid
    import json
    from typing import List, Dict, Any, Optional
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay
    import logging
    AIORTC_AVAILABLE = True
except ImportError as e:
    AIORTC_AVAILABLE = False
    st.warning(f"aiortc khu00f4ng khu1ea3 du1ee5ng: {str(e)}. Chu1ec9 su1eed du1ee5ng chu1ee9c nu0103ng upload u1ea3nh.")

# Thiu1ebft lu1eadp giao diu1ec7n trang
st.set_page_config(
    page_title="Hu1ec6 THu1ed0NG QUu1ea2N Lu00dd Cu00d4NG Du00c2N",
    page_icon="ud83dudccb",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cu1ea5u hu00ecnh logging cho aiortc
if AIORTC_AVAILABLE:
    logging.basicConfig(level=logging.INFO)

# Lu01b0u tru1eef cu00e1c ku1ebft nu1ed1i peer
peer_connections = {}
videoframes = {}

# Lu1edbp VideoProcessor cho aiortc
class VideoProcessor:
    def __init__(self, callback=None):
        self.callback = callback
        self.qr_detected = False
        self.qr_data = None
        
    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Nu1ebfu cu00f3 callback, gu1ecdi nu00f3 u0111u1ec3 xu1eed lu00fd frame
        if self.callback:
            img = self.callback(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Lu1edbp VideoStreamTrack tu00f9y chu1ec9nh
class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, processor=None):
        super().__init__()
        self.track = track
        self.processor = processor if processor else VideoProcessor()

    async def recv(self):
        frame = await self.track.recv()
        if self.processor:
            frame = self.processor.process(frame)
        return frame

# Lu1edbp QRCodeProcessor
class QRCodeProcessor(VideoProcessor):
    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Xu1eed lu00fd QR code
        try:
            # Chuyu1ec3n sang RGB u0111u1ec3 xu1eed lu00fd
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded_objects = decode(frame_rgb)
            
            for obj in decoded_objects:
                # Vu1ebd khung xung quanh QR code
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(img, [hull], True, (0, 255, 0), 2)
                else:
                    cv2.polylines(img, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
                
                # Giu1ea3i mu00e3 QR
                qr_data = obj.data.decode('utf-8')
                self.qr_data = qr_data
                self.qr_detected = True
                
                # Hiu1ec3n thu1ecb thu00f4ng tin
                cv2.putText(img, "QR Code Detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Error processing QR code: {str(e)}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# CSS tu00f9y chu1ec9nh
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #004d99;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .success-message {
        padding: 20px;
        background-color: #4CAF50;
        color: white;
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .error-message {
        padding: 20px;
        background-color: #f44336;
        color: white;
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .info-card {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .feature-button {
        background-color: #ffffff;
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .feature-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .camera-feed {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .login-form {
        max-width: 500px;
        margin: 0 auto;
        padding: 30px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .login-header {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Khu1edfi tu1ea1o session state
if 'citizens_data' not in st.session_state:
    st.session_state.citizens_data = pd.DataFrame(columns=[
        'id', 'cccd', 'name', 'dob', 'sex', 'address', 'expdate', 'scan_date', 'image_path'
    ])

# Thu00eam session state cho u0111u0103ng nhu1eadp
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

# Thu00eam session state cho u0111iu1ec1u hu01b0u1edbng trang
if 'page' not in st.session_state:
    st.session_state.page = None
    
# Thu00eam session state cho menu choice
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Trang chu1ee7"

# Thu00eam session state cho aiortc
if 'peer_connection_id' not in st.session_state:
    st.session_state.peer_connection_id = None

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None

# Danh su00e1ch tu00e0i khou1ea3n mu1eabu (trong thu1ef1c tu1ebf nu00ean lu01b0u trong cu01a1 su1edf du1eef liu1ec7u vu00e0 mu00e3 hu00f3a mu1eadt khu1ea9u)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

# ICE servers configuration for aiortc
ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]}
]

# Hu00e0m xu1eed lu00fd aiortc
async def process_offer(offer, video_processor=None):
    pc_id = str(uuid.uuid4())
    pc = RTCPeerConnection({"iceServers": ICE_SERVERS})
    peer_connections[pc_id] = pc
    
    relay = MediaRelay()
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Su1eed du1ee5ng video processor nu1ebfu cu00f3
            if video_processor:
                transformed_track = VideoTransformTrack(relay.subscribe(track), processor=video_processor)
            else:
                # Mu1eb7c u0111u1ecbnh chu1ec9 chuyu1ec3n tiu1ebfp video
                transformed_track = VideoTransformTrack(relay.subscribe(track))
                
            pc.addTrack(transformed_track)
    
    # Thiu1ebft lu1eadp ku1ebft nu1ed1i
    offer = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, pc_id

async def close_peer_connection(pc_id):
    if pc_id in peer_connections:
        pc = peer_connections[pc_id]
        await pc.close()
        del peer_connections[pc_id]

def setup_asyncio():
    """Setup asyncio event loop for WebRTC"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

@contextmanager
def aiortc_context():
    """Context manager for aiortc operations"""
    try:
        setup_asyncio()
        yield
    except Exception as e:
        st.error(f"aiortc Error: {str(e)}")
        st.info("Please try refreshing the page or use the image upload feature instead.")

def create_webrtc_component(key, video_processor=None):
    """Create a WebRTC component using aiortc"""
    if not AIORTC_AVAILABLE:
        st.error("aiortc khu00f4ng khu1ea3 du1ee5ng. Vui lu00f2ng su1eed du1ee5ng chu1ee9c nu0103ng upload u1ea3nh.")
        return None
    
    # Tu1ea1o container cho video
    video_container = st.empty()
    status_container = st.empty()
    
    # Tu1ea1o cu00e1c nu00fat u0111iu1ec1u khiu1ec3n
    col1, col2 = st.columns(2)
    start_button = col1.button("Bu1eaft u0111u1ea7u Camera", key=f"start_{key}")
    stop_button = col2.button("Du1eebng Camera", key=f"stop_{key}")
    
    # Xu1eed lu00fd khi nhu1ea5n nu00fat bu1eaft u0111u1ea7u
    if start_button:
        status_container.info("u0110ang ku1ebft nu1ed1i camera...")
        
        # Tu1ea1o offer SDP
        offer = {
            "sdp": "",
            "type": "offer"
        }
        
        # Xu1eed lu00fd offer bu1eb1ng asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answer, pc_id = loop.run_until_complete(process_offer(offer, video_processor))
        
        # Lu01b0u ID ku1ebft nu1ed1i
        st.session_state.peer_connection_id = pc_id
        st.session_state.video_processor = video_processor
        
        # Hiu1ec3n thu1ecb tru1ea1ng thu00e1i
        status_container.success("Camera u0111ang hou1ea1t u0111u1ed9ng")
        
        # Hiu1ec3n thu1ecb video (giu1ea3 lu1eadp)
        video_container.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB", use_container_width=True)
        
    # Xu1eed lu00fd khi nhu1ea5n nu00fat du1eebng
    if stop_button and st.session_state.peer_connection_id:
        # u0110u00f3ng ku1ebft nu1ed1i
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(close_peer_connection(st.session_state.peer_connection_id))
        
        # Xu00f3a ID ku1ebft nu1ed1i
        st.session_state.peer_connection_id = None
        st.session_state.video_processor = None
        
        # Hiu1ec3n thu1ecb tru1ea1ng thu00e1i
        status_container.info("Camera u0111u00e3 du1eebng")
        video_container.empty()
    
    return st.session_state.video_processor

def login_page():
    st.markdown("<h1 style='text-align: center;'>u0110u0103ng nhu1eadp Hu1ec7 thu1ed1ng</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-card" style="padding: 30px;">
            <h3 style="text-align: center;">u0110u0103ng nhu1eadp</h3>
            <p style="text-align: center;">Vui lu00f2ng nhu1eadp thu00f4ng tin u0111u0103ng nhu1eadp cu1ee7a bu1ea1n.</p>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Tu00ean u0111u0103ng nhu1eadp")
        password = st.text_input("Mu1eadt khu1ea9u", type="password")
        
        if st.button("u0110u0103ng nhu1eadp"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"u0110u0103ng nhu1eadp thu00e0nh cu00f4ng! Chu00e0o mu1eebng, {username}")
                st.rerun()
            else:
                st.error("Tu00ean u0111u0103ng nhu1eadp hou1eb7c mu1eadt khu1ea9u khu00f4ng u0111u00fang!")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    flipped = img[::-1,:,:]
    return av.VideoFrame.from_ndarray(flipped, format="bgr24")

# Create a processor for flipping video
class FlipVideoProcessor(VideoProcessor):
    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        flipped = img[::-1,:,:]
        return av.VideoFrame.from_ndarray(flipped, format="bgr24")

def surveillance_camera():
    st.markdown("<h2 style='text-align: center;'>Giu00e1m su00e1t Camera</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Giu00e1m su00e1t an ninh</h3>
        <p>Theo du00f5i vu00e0 phu00e1t hiu1ec7n u0111u1ed1i tu01b0u1ee3ng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose between aiortc and file upload
        camera_option = st.radio(
            "Chu1ecdn phu01b0u01a1ng thu1ee9c:",
            ["Camera tru1ef1c tiu1ebfp (aiortc)", "Upload video/u1ea3nh"],
            key="camera_option"
        )
        
        if camera_option == "Camera tru1ef1c tiu1ebfp (aiortc)":
            try:
                if AIORTC_AVAILABLE:
                    # Su1eed du1ee5ng aiortc
                    processor = FlipVideoProcessor()
                    video_processor = create_webrtc_component("surveillance", processor)
                    
                    # Display connection status
                    if st.session_state.peer_connection_id:
                        st.success("u2705 Camera u0111ang hou1ea1t u0111u1ed9ng")
                    else:
                        st.info("ud83dudcf7 Nhu1ea5n 'Bu1eaft u0111u1ea7u Camera' u0111u1ec3 bu1eaft u0111u1ea7u camera")
                else:
                    st.error("aiortc khu00f4ng khu1ea3 du1ee5ng. Vui lu00f2ng su1eed du1ee5ng chu1ee9c nu0103ng upload u1ea3nh.")
                    
            except Exception as e:
                st.error(f"Lu1ed7i ku1ebft nu1ed1i camera: {str(e)}")
                st.info("Vui lu00f2ng thu1eed su1eed du1ee5ng tu00f9y chu1ecdn 'Upload video/u1ea3nh' bu00ean du01b0u1edbi")
                
        else:
            # Alternative: File upload for surveillance
            uploaded_file = st.file_uploader(
                "Tu1ea3i lu00ean video hou1eb7c u1ea3nh u0111u1ec3 phu00e2n tu00edch",
                type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
                key="surveillance_upload"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption="u1ea2nh u0111u00e3 tu1ea3i lu00ean", use_container_width=True)
                    
                    if st.button("Phu00e2n tu00edch u1ea3nh"):
                        st.success("u0110ang phu00e2n tu00edch u1ea3nh...")
                        # Add your image analysis logic here
                        
                else:
                    st.video(uploaded_file)
                    if st.button("Phu00e2n tu00edch video"):
                        st.success("u0110ang phu00e2n tu00edch video...")
                        # Add your video analysis logic here

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>u0110iu1ec1u khiu1ec3n Camera</h3>
        <p>Cu00e0i u0111u1eb7t vu00e0 u0111iu1ec1u khiu1ec3n camera giu00e1m su00e1t</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        detection_options = st.multiselect(
            "Chu1ecdn cu00e1c u0111u1ed1i tu01b0u1ee3ng cu1ea7n phu00e1t hiu1ec7n:",
            ["Khuu00f4n mu1eb7t", "Phu01b0u01a1ng tiu1ec7n", "Vu1eadt thu1ec3 khu1ea3 nghi"],
            default=["Khuu00f4n mu1eb7t"],
            key="detection_options"
        )
        
        sensitivity = st.slider("u0110u1ed9 nhu1ea1y phu00e1t hiu1ec7n", 0, 100, 50, key="sensitivity")
        
        if st.button("Chu1ee5p u1ea3nh", key="capture_btn"):
            st.success("u1ea2nh u0111u00e3 u0111u01b0u1ee3c chu1ee5p thu00e0nh cu00f4ng!")
            
        # Add troubleshooting section
        with st.expander("ud83dudd27 Khu1eafc phu1ee5c su1ef1 cu1ed1"):
            st.markdown("""
            **Nu1ebfu camera khu00f4ng hou1ea1t u0111u1ed9ng:**
            1. Lu00e0m mu1edbi trang (F5)
            2. Kiu1ec3m tra quyu1ec1n truy cu1eadp camera
            3. Thu1eed su1eed du1ee5ng tru00ecnh duyu1ec7t khu00e1c
            4. Su1eed du1ee5ng tu00f9y chu1ecdn 'Upload video/u1ea3nh'
            """)

def process_image_for_qr(image):
    """
    Xu1eed lu00fd u1ea3nh u0111u1ec3 tu00ecm vu00e0 giu1ea3i mu00e3 QR code
    """
    try:
        # Chuyu1ec3n u0111u1ed5i u1ea3nh sang u0111u1ecbnh du1ea1ng phu00f9 hu1ee3p
        if isinstance(image, np.ndarray):
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.array(image)

        # Giu1ea3i mu00e3 QR
        decoded_objects = decode(frame_rgb)
        
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            citizen_info = qr_data.split('|')
            
            if len(citizen_info) >= 7:
                # Tu1ea1o thu01b0 mu1ee5c lu01b0u u1ea3nh nu1ebfu chu01b0a tu1ed3n tu1ea1i
                os.makedirs("uploaded_images", exist_ok=True)
                
                # Tu1ea1o tu00ean file u1ea3nh vu1edbi timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"citizen_image_{timestamp}.jpg"
                image_path = os.path.join("uploaded_images", image_filename)
                
                # Lu01b0u u1ea3nh
                if isinstance(image, np.ndarray):
                    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    image.save(image_path)
    
                # Tu1ea1o bu1ea3n ghi mu1edbi
                new_data = {
                    'id': citizen_info[0],
                    'cccd': citizen_info[1],
                    'name': citizen_info[2],
                    'dob': citizen_info[3],
                    'sex': citizen_info[4],
                    'address': citizen_info[5],
                    'expdate': citizen_info[6],
                    'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': image_path
                }
                
                # Cu1eadp nhu1eadt DataFrame
                st.session_state.citizens_data = pd.concat([
                    st.session_state.citizens_data,
                    pd.DataFrame([new_data])
                ], ignore_index=True)
                
                return True, "QR code processed successfully!"
                
        return False, "Lu1ed7i khu00f4ng xu00e1c u0111u1ecbnh."
    
    except Exception as e:
        return False, f"Lu1ed7i: {str(e)}"

def scan_qr_code():
    """Enhanced QR code scanning with better error handling"""
    st.markdown("<h2 style='text-align: center;'>Quu00e9t mu00e3 QR CCCD</h2>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["ud83dudcc1 Upload u1ea2nh", "ud83dudcf7 Camera"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
        <h3>Tu1ea3i lu00ean u1ea3nh QR Code</h3>
        <p>u0110u1ecbnh du1ea1ng hu1ed7 tru1ee3: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Chu1ecdn u1ea3nh chu1ee9a QR code", 
            type=['jpg', 'jpeg', 'png'],
            key="qr_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="u1ea2nh u0111u00e3 tu1ea3i lu00ean", use_container_width=True)
            
            if st.button("Xu1eed lu00fd QR Code", key="process_qr"):
                with st.spinner("u0110ang xu1eed lu00fd..."):
                    success, message = process_image_for_qr(image)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    with tab2:
        st.markdown("""
        <div class="info-card">
        <h3>Quu00e9t qua Camera</h3>
        <p>Su1eed du1ee5ng camera u0111u1ec3 quu00e9t QR code tru1ef1c tiu1ebfp</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose scanning method
        scan_method = st.radio(
            "Chu1ecdn phu01b0u01a1ng thu1ee9c quu00e9t:",
            ["Camera aiortc", "Chu1ee5p u1ea3nh tu1eeb camera"],
            key="scan_method"
        )
        
        if scan_method == "Camera aiortc":
            try:
                if AIORTC_AVAILABLE:
                    # Su1eed du1ee5ng aiortc vu1edbi QR processor
                    qr_processor = QRCodeProcessor()
                    video_processor = create_webrtc_component("qr-scanner", qr_processor)
                    
                    # Process QR detection results
                    if st.session_state.video_processor and hasattr(st.session_state.video_processor, 'qr_detected') and st.session_state.video_processor.qr_detected:
                        process_qr_detection(st.session_state.video_processor.qr_data)
                else:
                    st.error("aiortc khu00f4ng khu1ea3 du1ee5ng. Vui lu00f2ng su1eed du1ee5ng tu00f9y chu1ecdn upload u1ea3nh.")
                        
            except Exception as e:
                st.error(f"Lu1ed7i camera: {str(e)}")
                st.info("Vui lu00f2ng thu1eed su1eed du1ee5ng tu00f9y chu1ecdn 'Chu1ee5p u1ea3nh tu1eeb camera' hou1eb7c 'Upload u1ea2nh'")
        
        else:
            # Alternative camera capture method
            if st.button("Chu1ee5p u1ea3nh tu1eeb camera", key="capture_camera"):
                st.info("Chu1ee9c nu0103ng nu00e0y u0111ang u0111u01b0u1ee3c phu00e1t triu1ec3n. Vui lu00f2ng su1eed du1ee5ng tu00f9y chu1ecdn upload u1ea3nh.")

def process_qr_detection(qr_data):
    """Process detected QR code data"""
    try:
        citizen_info = qr_data.split('|')
        
        if len(citizen_info) >= 7:
            st.success("u2705 QR code u0111u00e3 u0111u01b0u1ee3c phu00e1t hiu1ec7n vu00e0 xu1eed lu00fd thu00e0nh cu00f4ng!")
            
            # Save information to DataFrame
            new_data = {
                'id': citizen_info[0],
                'cccd': citizen_info[1],
                'name': citizen_info[2],
                'dob': citizen_info[3],
                'sex': citizen_info[4],
                'address': citizen_info[5],
                'expdate': citizen_info[6],
                'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_path': "camera_capture"
            }
            
            st.session_state.citizens_data = pd.concat([
                st.session_state.citizens_data,
                pd.DataFrame([new_data])
            ], ignore_index=True)
            
            # Display citizen information
            display_citizen_info(citizen_info)
            
    except Exception as e:
        st.error(f"Lu1ed7i xu1eed lu00fd QR code: {str(e)}")

def display_citizen_info(citizen_info):
    """Display citizen information in a formatted way"""
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h4 style="color: #2e7d32;">Thu00f4ng tin cu00f4ng du00e2n:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ID:** {citizen_info[0]}")
        st.write(f"**Su1ed1 CCCD:** {citizen_info[1]}")
        st.write(f"**Hu1ecd vu00e0 tu00ean:** {citizen_info[2]}")
        st.write(f"**Ngu00e0y sinh:** {citizen_info[3]}")
    
    with col2:
        st.write(f"**Giu1edbi tu00ednh:** {citizen_info[4]}")
        st.write(f"**u0110u1ecba chu1ec9:** {citizen_info[5]}")
        st.write(f"**Ngu00e0y hu1ebft hu1ea1n:** {citizen_info[6]}")

# Add this to the top of your main() function
def reset_session_on_error():
    """Reset session state if there are WebRTC errors"""
    if 'aiortc_error_count' not in st.session_state:
        st.session_state.aiortc_error_count = 0
    
    if st.session_state.aiortc_error_count > 3:
        st.warning("Phu00e1t hiu1ec7n nhiu1ec1u lu1ed7i aiortc. u0110ang reset session...")
        for key in list(st.session_state.keys()):
            if 'aiortc' in key.lower() or 'peer' in key.lower():
                del st.session_state[key]
        st.session_state.aiortc_error_count = 0
        st.rerun()

def show_citizen_data():
    st.markdown("<h2 style='text-align: center;'>Du1eef liu1ec7u Cu00f4ng du00e2n</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Cu00f4ng du00e2n:{row['name']} - CCCD: {row['cccd']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="u1ea3nh CCCD", use_container_width=True)
                    else:
                        st.warning("u1ea2nh CCCD khu00f4ng tu1ed3n tu1ea1i!")
                
                with col2:
                    st.markdown(f"""
                    **ID:** {row['id']}  
                    **Su1ed1 CCCD:** {row['cccd']}  
                    **Tu00ean:** {row['name']}  
                    **Ngu00e0y sinh:** {row['dob']}  
                    **Giu1edbi tu00ednh:** {row['sex']}  
                    **u0110u1ecba chu1ec9:** {row['address']}  
                    **Ngu00e0y hu1ebft hu1ea1n:** {row['expdate']}  
                    **Ngu00e0y quu00e9t:** {row['scan_date']}
                    """)
    else:
        st.info("Chu01b0a cu00f3 du1eef liu1ec7u cu00f4ng du00e2n nu00e0o.")


def show_homepage():
    st.markdown("<h1 style='text-align: center;'>Hu1ec7 thu1ed1ng Quu1ea3n lu00fd Cu00f4ng du00e2n</h1>", unsafe_allow_html=True)
    
    # Grid layout cho cu00e1c chu1ee9c nu0103ng
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-button">
            <h3>Quu00e9t QR CCCD</h3>
            <p>Quu00e9t mu00e3 QR tu1eeb CCCD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng quu00e9t QR
        if st.button("Quu00e9t QR CCCD"):
            st.session_state.page = "scan_qr"
            st.session_state.menu_choice = "Quu00e9t QR CCCD"
            st.rerun()
        
    with col2:
        st.markdown("""
        <div class="feature-button">
            <h3>Quu1ea3n lu00fd Cu00f4ng du00e2n</h3>
            <p>Quu1ea3n lu00fd du1eef liu1ec7u cu00f4ng du00e2n hiu1ec7u quu1ea3 vu00e0 du1ec5 du00e0ng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng quu1ea3n lu00fd du1eef liu1ec7u
        if st.button("Xem du1eef liu1ec7u cu00f4ng du00e2n"):
            st.session_state.page = "view_data"
            st.session_state.menu_choice = "Xem du1eef liu1ec7u"
            st.rerun()
        
    with col3:
        st.markdown("""
        <div class="feature-button">
            <h3>Camera Giu00e1m Su00e1t</h3>
            <p>Theo du00f5i qua camera an ninh</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng camera giu00e1m su00e1t
        if st.button("Camera giu00e1m su00e1t"):
            st.session_state.page = "camera"
            st.session_state.menu_choice = "Camera Giu00e1m su00e1t"
            st.rerun()
    
    # Kiu1ec3m tra nu1ebfu cu00f3 chuyu1ec3n trang tu1eeb cu00e1c nu00fat
    if 'page' in st.session_state:
        if st.session_state.page == "scan_qr":
            scan_qr_code()
            st.session_state.page = None
        elif st.session_state.page == "view_data":
            show_citizen_data()
            st.session_state.page = None
        elif st.session_state.page == "camera":
            surveillance_camera()
            st.session_state.page = None

def show_statistics():
    st.markdown("<h2 style='text-align: center;'>Thu1ed1ng ku00ea</h2>", unsafe_allow_html=True)
    st.write("Hiu1ec3n thu1ecb cu00e1c su1ed1 liu1ec7u thu1ed1ng ku00ea liu00ean quan u0111u1ebfn cu00f4ng du00e2n.")
    # Thu00eam code hiu1ec3n thu1ecb thu1ed1ng ku00ea

def show_settings():
    st.markdown("<h2 style='text-align: center;'>Cu00e0i u0111u1eb7t</h2>", unsafe_allow_html=True)
    st.write("Tu00f9y chu1ec9nh cu00e1c thiu1ebft lu1eadp cu1ee7a hu1ec7 thu1ed1ng tu1ea1i u0111u00e2y.")
    # Thu00eam code cu00e0i u0111u1eb7t


def main():
    # Kiu1ec3m tra u0111u0103ng nhu1eadp
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Hiu1ec3n thu1ecb giao diu1ec7n chu00ednh sau khi u0111u0103ng nhu1eadp
    st.sidebar.markdown("<h1 style='text-align: center;'>Chu00e0o mu1eebng ud83dudcf7</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center;'>Quu1ea3n lu00fd Cu00f4ng du00e2n</h2>", unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb thu00f4ng tin ngu01b0u1eddi du00f9ng u0111u0103ng nhu1eadp
    st.sidebar.markdown(f"""<div style='text-align: center; padding: 10px; background-color: #e8f5e9; 
                        border-radius: 5px; margin-bottom: 20px;'>
                         <b>{st.session_state.username}</b></div>""", 
                        unsafe_allow_html=True)
    
    menu = [
        "Trang chu1ee7",
        "Quu00e9t QR CCCD",
        "Xem du1eef liu1ec7u",
        "Camera Giu00e1m su00e1t",
        "Thu1ed1ng ku00ea",
        "Cu00e0i u0111u1eb7t"
    ]
    
    choice = st.sidebar.selectbox(
        "Chu1ecdn chu1ee9c nu0103ng", 
        menu, 
        index=menu.index(st.session_state.menu_choice),
        key="main_menu"
    )
    
    # Hiu1ec3n thu1ecb cu00e1c nu00fat chu1ee9c nu0103ng phu1ee5 trong sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Chu1ee9c nu0103ng nhanh")
    
    if st.sidebar.button("ud83dudcf7 Camera"):
        st.session_state.page = "camera"
        st.session_state.menu_choice = "Camera Giu00e1m su00e1t"
        st.rerun()
      
    if st.sidebar.button("ud83dudcca Bu00e1o cu00e1o"):
        st.session_state.page = "reports"
        st.session_state.menu_choice = "Thu1ed1ng ku00ea"
        st.rerun()
        
    if st.sidebar.button("u2699ufe0f Cu00e0i u0111u1eb7t"):
        st.session_state.page = "settings"
        st.session_state.menu_choice = "Cu00e0i u0111u1eb7t"
        st.rerun()
    
    # Nu00fat u0111u0103ng xuu1ea5t
    if st.sidebar.button("ud83dudeaa u0110u0103ng xuu1ea5t"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if choice == "Trang chu1ee7":
        show_homepage()
    elif choice == "Quu00e9t QR CCCD":
        scan_qr_code()
    elif choice == "Xem du1eef liu1ec7u":
        show_citizen_data()
    elif choice == "Camera Giu00e1m su00e1t":
        surveillance_camera()
    elif choice == "Thu1ed1ng ku00ea":
        show_statistics()
    elif choice == "Cu00e0i u0111u1eb7t":
        show_settings()

if __name__ == '__main__':
    main()