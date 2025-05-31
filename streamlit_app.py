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
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
# Thêm try-except cho import asyncio để xử lý lỗi liên quan đến asyncio
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
    st.warning(f"aiortc không khả dụng: {str(e)}. Chỉ sử dụng chức năng upload ảnh.")

# Thiết lập giao diện trang
st.set_page_config(
    page_title="HỆ THỐNG QUẢN LÝ CÔNG DÂN",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình logging cho aiortc
if AIORTC_AVAILABLE:
    logging.basicConfig(level=logging.INFO)

# Lưu trữ các kết nối peer
peer_connections = {}
videoframes = {}
class SafeRTCConfiguration:
    """Safe RTC configuration with error handling"""
    
    def __init__(self):
        self.ice_servers = []
        self.connection_timeout = 30
        self.gathering_timeout = 10
        self.retry_attempts = 3
    
    def get_safe_rtc_config(self) -> RTCConfiguration:
        """Get RTC configuration with error handling"""
        
        # Basic STUN servers (always working)
        safe_ice_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
        
        # Test additional servers
        additional_servers = [
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
        
        for server in additional_servers:
            if self._test_stun_server(server["urls"][0]):
                safe_ice_servers.append(server)
        
        # Add TURN servers if available
        turn_servers = self._get_working_turn_servers()
        if turn_servers:
            safe_ice_servers.extend(turn_servers)
        
        return RTCConfiguration({
            "iceServers": safe_ice_servers,
            "iceConnectionTimeout": self.connection_timeout,
            "iceGatheringTimeout": self.gathering_timeout,
            "bundlePolicy": "balanced",
            "rtcpMuxPolicy": "require"
        })
    
    def _test_stun_server(self, stun_url: str, timeout: float = 2.0) -> bool:
        """Test if STUN server is reachable"""
        try:
            # Parse STUN URL
            if not stun_url.startswith("stun:"):
                return False
            
            host_port = stun_url[5:]  # Remove "stun:"
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                port = int(port)
            else:
                host = host_port
                port = 3478
            
            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            
            try:
                sock.connect((host, port))
                return True
            except:
                return False
            finally:
                sock.close()
                
        except Exception as e:
            logger.warning(f"STUN server test failed for {stun_url}: {e}")
            return False
    
    def _get_working_turn_servers(self) -> List[Dict]:
        """Get working TURN servers"""
        turn_servers = []
        
        try:
            # Try Twilio TURN if configured
            if hasattr(st.secrets, 'webrtc') and 'twilio_account_sid' in st.secrets.webrtc:
                twilio_servers = self._get_twilio_turn_safe()
                if twilio_servers:
                    turn_servers.extend(twilio_servers)
            
            # Try other TURN providers
            # Add your TURN server configurations here
            
        except Exception as e:
            logger.warning(f"TURN server setup failed: {e}")
        
        return turn_servers
    
    def _get_twilio_turn_safe(self) -> List[Dict]:
        """Safely get Twilio TURN servers"""
        try:
            import requests
            
            account_sid = st.secrets.webrtc.twilio_account_sid
            auth_token = st.secrets.webrtc.twilio_auth_token
            
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
            
            response = requests.post(
                url, 
                auth=(account_sid, auth_token),
                timeout=5  # Short timeout
            )
            
            if response.status_code == 201:
                token_data = response.json()
                return token_data.get("ice_servers", [])
            
        except Exception as e:
            logger.warning(f"Twilio TURN failed: {e}")
        
        return []

class ObjectDetectionTransformer(VideoProcessorBase):
    def recv(self, frame):
       
        img = frame.to_ndarray(format="bgr24")
        
        # Thu00eam logic phu00e1t hiu1ec7n u0111u1ed1i tu01b0u1ee3ng u1edf u0111u00e2y
        # (Cu00f3 thu1ec3 su1eed du1ee5ng OpenCV, YOLO, hou1eb7c cu00e1c model khu00e1c)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
# Lớp VideoProcessor cho aiortc
class VideoProcessor:
    def __init__(self, callback=None):
        self.callback = callback
        self.qr_detected = False
        self.qr_data = None
        
    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Nếu có callback, gọi nó để xử lý frame
        if self.callback:
            img = self.callback(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Lớp VideoStreamTrack tùy chỉnh
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

# Lớp QRCodeProcessor
class QRCodeProcessor(VideoProcessor):
    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Xử lý QR code
        try:
            # Chuyển sang RGB để xử lý
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded_objects = decode(frame_rgb)
            
            for obj in decoded_objects:
                # Vẽ khung xung quanh QR code
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(img, [hull], True, (0, 255, 0), 2)
                else:
                    cv2.polylines(img, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
                
                # Giải mã QR
                qr_data = obj.data.decode('utf-8')
                self.qr_data = qr_data
                self.qr_detected = True
                
                # Hiển thị thông tin
                cv2.putText(img, "QR Code Detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Error processing QR code: {str(e)}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# CSS tùy chỉnh
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


# Khởi tạo session state
if 'citizens_data' not in st.session_state:
    st.session_state.citizens_data = pd.DataFrame(columns=[
        'id', 'cccd', 'name', 'dob', 'sex', 'address', 'expdate', 'scan_date', 'image_path'
    ])

# Thêm session state cho đăng nhập
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

# Thêm session state cho điều hướng trang
if 'page' not in st.session_state:
    st.session_state.page = None
    
# Thêm session state cho menu choice
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Trang chủ"

# Thêm session state cho aiortc
if 'peer_connection_id' not in st.session_state:
    st.session_state.peer_connection_id = None

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None

# Danh sách tài khoản mẫu (trong thực tế nên lưu trong cơ sở dữ liệu và mã hóa mật khẩu)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

# ICE servers configuration for aiortc
ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]}
]

# Hàm xử lý aiortc
async def process_offer(offer, video_processor=None):
    #offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc_id = str(uuid.uuid4())
    pc = RTCPeerConnection()
    peer_connections[pc_id] = pc
    
    relay = MediaRelay()
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Sử dụng video processor nếu có
            if video_processor:
                transformed_track = VideoTransformTrack(relay.subscribe(track), processor=video_processor)
            else:
                # Mặc định chỉ chuyển tiếp video
                transformed_track = VideoTransformTrack(relay.subscribe(track))
                
            pc.addTrack(transformed_track)
    
    # Thiết lập kết nối
    if offer and offer.get("sdp"):
        offer = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Lưu ID kết nối
        st.session_state.peer_connection_id = pc_id
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, pc_id
    else:
        # Nếu không có offer, tạo một local stream
        local_video = VideoTransformTrack(MediaPlayer('default:none', format='bgr24').video, processor=video_processor)
        pc.addTrack(local_video)
        
        # Tạo offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Lưu ID kết nối
        st.session_state.peer_connection_id = pc_id
        
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
        st.error("aiortc không khả dụng. Vui lòng sử dụng chức năng upload ảnh.")
        return None
    
    # Tạo container cho video
    video_container = st.empty()
    status_container = st.empty()
    
    # Tạo các nút điều khiển
    col1, col2 = st.columns(2)
    start_button = col1.button("Bắt đầu Camera", key=f"start_{key}")
    stop_button = col2.button("Dừng Camera", key=f"stop_{key}")
    
    # Tạo JavaScript để truy cập camera
    js_code = """
    <script>
    const getWebcamVideo = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            const videoTracks = stream.getVideoTracks();
            const track = videoTracks[0];
            
            // Create peer connection
            const pc = new RTCPeerConnection({
                iceServers: [{urls: ['stun:stun.l.google.com:19302']}]
            });
            
            // Add track to peer connection
            pc.addTrack(track, stream);
            
            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Send offer to server
            const response = await fetch('/_stcore/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: 'webrtc_offer',
                    sdp: pc.localDescription.sdp,
                    session_id: '%s'
                }),
            });
            
            const data = await response.json();
            const answer = new RTCSessionDescription({
                type: 'answer',
                sdp: data.sdp
            });
            
            await pc.setRemoteDescription(answer);
            
            return pc;
        } catch (err) {
            console.error('Error accessing webcam:', err);
            return null;
        }
    };
    
    // Start webcam when button is clicked
    const startButton = document.querySelector('button[data-testid="start_%s"]');
    if (startButton) {
        startButton.addEventListener('click', () => {
            getWebcamVideo().then(pc => {
                window.webrtcPc = pc;
            });
        });
    }
    
    // Stop webcam when button is clicked
    const stopButton = document.querySelector('button[data-testid="stop_%s"]');
    if (stopButton) {
        stopButton.addEventListener('click', () => {
            if (window.webrtcPc) {
                window.webrtcPc.close();
                window.webrtcPc = null;
            }
        });
    }
    </script>
    """ % (key, key, key)
    
    # Inject JavaScript
    st.markdown(js_code, unsafe_allow_html=True)
    
    # Xử lý khi nhấn nút bắt đầu
    if start_button:
        status_container.info("Đang kết nối camera...")
        
        # Tạo placeholder cho video stream
        video_container.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB", use_container_width=True)
        
        # Lưu processor
        st.session_state.video_processor = video_processor
        
        # Hiển thị trạng thái
        status_container.success("Camera đang hoạt động")
        
    # Xử lý khi nhấn nút dừng
    if stop_button and st.session_state.peer_connection_id:
        # Đóng kết nối nếu có
        if st.session_state.peer_connection_id:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(close_peer_connection(st.session_state.peer_connection_id))
            
            # Xóa ID kết nối
            st.session_state.peer_connection_id = None
            st.session_state.video_processor = None
        
        # Hiển thị trạng thái
        status_container.info("Camera đã dừng")
        video_container.empty()
    
    return st.session_state.video_processor

def login_page():
    st.markdown("<h1 style='text-align: center;'>Đăng nhập Hệ thống</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-card" style="padding: 30px;">
            <h3 style="text-align: center;">Đăng nhập</h3>
            <p style="text-align: center;">Vui lòng nhập thông tin đăng nhập của bạn.</p>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        
        if st.button("Đăng nhập"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Đăng nhập thành công! Chào mừng, {username}")
                st.rerun()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không đúng!")

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
rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com:3478"]},
                {"urls": ["stun:stun5.l.google.com:19302"]},
                {"urls": ["stun:stun.cloudflare.com:3478"]},
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })
def get_ice_connection_fix_config():
    """Get RTC configuration specifically designed to fix ICE connection issues"""
    
    # Multiple STUN servers for redundancy
    stun_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        {"urls": ["stun:stun.services.mozilla.com:3478"]},
    ]
    
    # Free TURN servers for NAT traversal
    turn_servers = [
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject", 
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
    
    # Twilio TURN servers (if configured)
   
    
    all_ice_servers = stun_servers + turn_servers
    
    # Optimized RTC configuration for ICE connection issues
    rtc_config = RTCConfiguration({
        "iceServers": all_ice_servers,
        "iceTransportPolicy": "all",  # Use both STUN and TURN
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require",
        "iceCandidatePoolSize": 20,  # Increase candidate pool
        "iceConnectionTimeout": 45000,  # 45 seconds timeout
        "iceGatheringTimeout": 20000,   # 20 seconds gathering
        "continualGatheringPolicy": "gather_continually"  # Keep gathering
    })
    
    return rtc_config
flip = st.checkbox("Flip")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:] if flip else img

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")

    return frame
def surveillance_camera():
    st.markdown("<h2 style='text-align: center;'>Giám sát Camera</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Giám sát an ninh</h3>
        <p>Theo dõi và phát hiện đối tượng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose between aiortc and file upload
        camera_option = st.radio(
            "Chọn phương thức:",
            ["Camera trực tiếp (aiortc)", "Upload video/ảnh"],
            key="camera_option"
        )
        
        if camera_option == "Camera trực tiếp (aiortc)":
            try:
                if AIORTC_AVAILABLE:
                    safe_rtc = SafeRTCConfiguration()
                    rtc_config = safe_rtc.get_safe_rtc_config() 
                    # Sử dụng aiortc
                    processor = FlipVideoProcessor()
                    webrtc_ctx = webrtc_streamer(
                    key="camera-stream",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=get_ice_connection_fix_config(),
                    video_processor_factory=ObjectDetectionTransformer,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={
                        "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"min": 10, "ideal": 15, "max": 30}  # Lower FPS for stability
                },
                        "audio": False
                    },
                    async_processing=False,
                    sendback_audio=False,
                )
                    
                    # Display connection status
                    if st.session_state.peer_connection_id:
                        st.success("✅ Camera đang hoạt động")
                    else:
                        st.info("📷 Nhấn 'Bắt đầu Camera' để bắt đầu camera")
                else:
                    st.error("aiortc không khả dụng. Vui lòng sử dụng chức năng upload ảnh.")
                    
            except Exception as e:
                st.error(f"Lỗi kết nối camera: {str(e)}")
                st.info("Vui lòng thử sử dụng tùy chọn 'Upload video/ảnh' bên dưới")
                
        else:
            # Alternative: File upload for surveillance
            uploaded_file = st.file_uploader(
                "Tải lên video hoặc ảnh để phân tích",
                type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
                key="surveillance_upload"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
                    
                    if st.button("Phân tích ảnh"):
                        st.success("Đang phân tích ảnh...")
                        # Add your image analysis logic here
                        
                else:
                    st.video(uploaded_file)
                    if st.button("Phân tích video"):
                        st.success("Đang phân tích video...")
                        # Add your video analysis logic here

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Điều khiển Camera</h3>
        <p>Cài đặt và điều khiển camera giám sát</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        detection_options = st.multiselect(
            "Chọn các đối tượng cần phát hiện:",
            ["Khuôn mặt", "Phương tiện", "Vật thể khả nghi"],
            default=["Khuôn mặt"],
            key="detection_options"
        )
        
        sensitivity = st.slider("Độ nhạy phát hiện", 0, 100, 50, key="sensitivity")
        
        if st.button("Chụp ảnh", key="capture_btn"):
            st.success("Ảnh đã được chụp thành công!")
            
        # Add troubleshooting section
        with st.expander("🔧 Khắc phục sự cố"):
            st.markdown("""
            **Nếu camera không hoạt động:**
            1. Làm mới trang (F5)
            2. Kiểm tra quyền truy cập camera
            3. Thử sử dụng trình duyệt khác
            4. Sử dụng tùy chọn 'Upload video/ảnh'
            """)

def process_image_for_qr(image):
    """
    Xử lý ảnh để tìm và giải mã QR code
    """
    try:
        # Chuyển đổi ảnh sang định dạng phù hợp
        if isinstance(image, np.ndarray):
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.array(image)

        # Giải mã QR
        decoded_objects = decode(frame_rgb)
        
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            citizen_info = qr_data.split('|')
            
            if len(citizen_info) >= 7:
                # Tạo thư mục lưu ảnh nếu chưa tồn tại
                os.makedirs("uploaded_images", exist_ok=True)
                
                # Tạo tên file ảnh với timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"citizen_image_{timestamp}.jpg"
                image_path = os.path.join("uploaded_images", image_filename)
                
                # Lưu ảnh
                if isinstance(image, np.ndarray):
                    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    image.save(image_path)
    
                # Tạo bản ghi mới
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
                
                # Cập nhật DataFrame
                st.session_state.citizens_data = pd.concat([
                    st.session_state.citizens_data,
                    pd.DataFrame([new_data])
                ], ignore_index=True)
                
                return True, "QR code processed successfully!"
                
        return False, "Lỗi không xác định."
    
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

def scan_qr_code():
    """Enhanced QR code scanning with better error handling"""
    st.markdown("<h2 style='text-align: center;'>Quét mã QR CCCD</h2>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Ảnh", "📷 Camera"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
        <h3>Tải lên ảnh QR Code</h3>
        <p>Định dạng hỗ trợ: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh chứa QR code", 
            type=['jpg', 'jpeg', 'png'],
            key="qr_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            
            if st.button("Xử lý QR Code", key="process_qr"):
                with st.spinner("Đang xử lý..."):
                    success, message = process_image_for_qr(image)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    with tab2:
        st.markdown("""
        <div class="info-card">
        <h3>Quét qua Camera</h3>
        <p>Sử dụng camera để quét QR code trực tiếp</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use Streamlit's native camera input
        camera_image = st.camera_input("Quét mã QR", key="qr_camera")
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            image_array = np.array(image)
            
            # Display the image
            st.image(image_array, caption="Ảnh đã chụp", use_container_width=True)
            
            # Process for QR code
            if st.button("Xử lý QR Code", key="process_camera_qr"):
                with st.spinner("Đang xử lý..."):
                    success, message = process_image_for_qr(image)
                    if success:
                        st.success(message)
                    else:
                        st.error("Không tìm thấy mã QR trong ảnh. Vui lòng thử lại.")

def process_qr_detection(qr_data):
    """Process detected QR code data"""
    try:
        citizen_info = qr_data.split('|')
        
        if len(citizen_info) >= 7:
            st.success("✅ QR code đã được phát hiện và xử lý thành công!")
            
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
        st.error(f"Lỗi xử lý QR code: {str(e)}")

def display_citizen_info(citizen_info):
    """Display citizen information in a formatted way"""
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h4 style="color: #2e7d32;">Thông tin công dân:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ID:** {citizen_info[0]}")
        st.write(f"**Số CCCD:** {citizen_info[1]}")
        st.write(f"**Họ và tên:** {citizen_info[2]}")
        st.write(f"**Ngày sinh:** {citizen_info[3]}")
    
    with col2:
        st.write(f"**Giới tính:** {citizen_info[4]}")
        st.write(f"**Địa chỉ:** {citizen_info[5]}")
        st.write(f"**Ngày hết hạn:** {citizen_info[6]}")

# Add this to the top of your main() function
def reset_session_on_error():
    """Reset session state if there are WebRTC errors"""
    if 'aiortc_error_count' not in st.session_state:
        st.session_state.aiortc_error_count = 0
    
    if st.session_state.aiortc_error_count > 3:
        st.warning("Phát hiện nhiều lỗi aiortc. Đang reset session...")
        for key in list(st.session_state.keys()):
            if 'aiortc' in key.lower() or 'peer' in key.lower():
                del st.session_state[key]
        st.session_state.aiortc_error_count = 0
        st.rerun()

def show_citizen_data():
    st.markdown("<h2 style='text-align: center;'>Dữ liệu Công dân</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Công dân:{row['name']} - CCCD: {row['cccd']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="ảnh CCCD", use_container_width=True)
                    else:
                        st.warning("Ảnh CCCD không tồn tại!")
                
                with col2:
                    st.markdown(f"""
                    **ID:** {row['id']}  
                    **Số CCCD:** {row['cccd']}  
                    **Tên:** {row['name']}  
                    **Ngày sinh:** {row['dob']}  
                    **Giới tính:** {row['sex']}  
                    **Địa chỉ:** {row['address']}  
                    **Ngày hết hạn:** {row['expdate']}  
                    **Ngày quét:** {row['scan_date']}
                    """)
    else:
        st.info("Chưa có dữ liệu công dân nào.")


def show_homepage():
    st.markdown("<h1 style='text-align: center;'>Hệ thống Quản lý Công dân</h1>", unsafe_allow_html=True)
    
    # Grid layout cho các chức năng
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-button">
            <h3>Quét QR CCCD</h3>
            <p>Quét mã QR từ CCCD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút kết nối với chức năng quét QR
        if st.button("Quét QR CCCD"):
            st.session_state.page = "scan_qr"
            st.session_state.menu_choice = "Quét QR CCCD"
            st.rerun()
        
    with col2:
        st.markdown("""
        <div class="feature-button">
            <h3>Quản lý Công dân</h3>
            <p>Quản lý dữ liệu công dân hiệu quả và dễ dàng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút kết nối với chức năng quản lý dữ liệu
        if st.button("Xem dữ liệu công dân"):
            st.session_state.page = "view_data"
            st.session_state.menu_choice = "Xem dữ liệu"
            st.rerun()
        
    with col3:
        st.markdown("""
        <div class="feature-button">
            <h3>Camera Giám Sát</h3>
            <p>Theo dõi qua camera an ninh</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút kết nối với chức năng camera giám sát
        if st.button("Camera giám sát"):
            st.session_state.page = "camera"
            st.session_state.menu_choice = "Camera Giám sát"
            st.rerun()
    
    # Kiểm tra nếu có chuyển trang từ các nút
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
    st.markdown("<h2 style='text-align: center;'>Thống kê</h2>", unsafe_allow_html=True)
    st.write("Hiển thị các số liệu thống kê liên quan đến công dân.")
    # Thêm code hiển thị thống kê

def show_settings():
    st.markdown("<h2 style='text-align: center;'>Cài đặt</h2>", unsafe_allow_html=True)
    st.write("Tùy chỉnh các thiết lập của hệ thống tại đây.")
    # Thêm code cài đặt
from pathlib import Path

# Cloud environment detection
def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud"""
    return (
        os.getenv("STREAMLIT_SHARING_MODE") == "true" or
        "streamlit.io" in os.getenv("HOSTNAME", "") or
        os.path.exists("/.streamlit")
    )
def setup_cloud_environment():
    """Setup environment cho Streamlit Cloud"""
    
    if is_streamlit_cloud():
        st.sidebar.info("🌐 Running on Streamlit Cloud")
        
        # Cloud-specific configurations
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Disable MSMF on Windows
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        
        # Temporary directories
        temp_dirs = ["/tmp/uploads", "/tmp/rtc_cache", "/tmp/video_processing"]
        for temp_dir in temp_dirs:
            Path(temp_dir).mkdir(exist_ok=True)
        
        return True
    else:
        st.sidebar.info("💻 Running locally")
        return False
def main():
    # Kiểm tra đăng nhập
    if not st.session_state.logged_in:
        login_page()
        return
    is_cloud = setup_cloud_environment()
    if is_cloud:
        st.info("""
        🌐 **Cloud Deployment Notes:**
        - HTTPS enabled automatically
        - Camera access requires user permission
        - File uploads limited to 200MB
        - Temporary files auto-cleaned
        """)
    # Hiển thị giao diện chính sau khi đăng nhập
    st.sidebar.markdown("<h1 style='text-align: center;'>Chào mừng 📷</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center;'>Quản lý Công dân</h2>", unsafe_allow_html=True)
    
    # Hiển thị thông tin người dùng đăng nhập
    st.sidebar.markdown(f"""<div style='text-align: center; padding: 10px; background-color: #e8f5e9; 
                        border-radius: 5px; margin-bottom: 20px;'>
                         <b>{st.session_state.username}</b></div>""", 
                        unsafe_allow_html=True)
    
    menu = [
        "Trang chủ",
        "Quét QR CCCD",
        "Xem dữ liệu",
        "Camera Giám sát",
        "Thống kê",
        "Cài đặt"
    ]
    
    choice = st.sidebar.selectbox(
        "Chọn chức năng", 
        menu, 
        index=menu.index(st.session_state.menu_choice),
        key="main_menu"
    )
    
    # Hiển thị các nút chức năng phụ trong sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Chức năng nhanh")
    
    if st.sidebar.button("📷 Camera"):
        st.session_state.page = "camera"
        st.session_state.menu_choice = "Camera Giám sát"
        st.rerun()
      
    if st.sidebar.button("📊 Báo cáo"):
        st.session_state.page = "reports"
        st.session_state.menu_choice = "Thống kê"
        st.rerun()
        
    if st.sidebar.button("⚙️ Cài đặt"):
        st.session_state.page = "settings"
        st.session_state.menu_choice = "Cài đặt"
        st.rerun()
    
    # Nút đăng xuất
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if choice == "Trang chủ":
        show_homepage()
    elif choice == "Quét QR CCCD":
        scan_qr_code()
    elif choice == "Xem dữ liệu":
        show_citizen_data()
    elif choice == "Camera Giám sát":
        surveillance_camera()
    elif choice == "Thống kê":
        show_statistics()
    elif choice == "Cài đặt":
        show_settings()

if __name__ == '__main__':
    main()