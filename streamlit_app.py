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
from facenet_pytorch import MTCNN, InceptionResnetV1
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
class ObjectDetectionTransformer(VideoProcessorBase):
    def __init__(self):
        # Khởi tạo MTCNN cho phát hiện khuôn mặt
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # Ngưỡng phát hiện ba bước
            factor=0.709, 
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Khởi tạo FaceNet model (tùy chọn nếu bạn muốn nhận dạng khuôn mặt)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.facenet = self.facenet.cuda()
            
        # Biến để lưu trữ embedding khuôn mặt (tùy chọn)
        self.known_face_embeddings = []
        self.known_face_names = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Chuyển đổi từ BGR sang RGB (MTCNN sử dụng RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt bằng MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(rgb_img, landmarks=True)
        
        # Vẽ các khuôn mặt được phát hiện
        if boxes is not None:
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                # Lấy tọa độ khuôn mặt
                x1, y1, x2, y2 = [int(p) for p in box]
                
                # Vẽ hình chữ nhật xung quanh khuôn mặt
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ các điểm landmark (mắt, mũi, miệng)
                for p in landmark:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                
                # Hiển thị xác suất phát hiện
                confidence = f"Confidence: {probs[i]:.2f}"
                cv2.putText(img, confidence, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Tùy chọn: Trích xuất embedding khuôn mặt (nếu bạn muốn nhận dạng)
                # face = self.mtcnn(rgb_img[y1:y2, x1:x2])
                # if face is not None:
                #     with torch.no_grad():
                #         embedding = self.facenet(face.unsqueeze(0))
                #         # Ở đây bạn có thể so sánh embedding với các khuôn mặt đã biết
        
        # Hiển thị số lượng khuôn mặt được phát hiện
        if boxes is not None:
            face_count = len(boxes)
            cv2.putText(img, f"Faces: {face_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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



 
import requests
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
                    response = requests.get(
                        "https://iewcom.metered.live/api/v1/turn/credentials",
                        params={"apiKey": "5b0cc93867e02c9b2e8ef46de385169008aa"}
                    )
                    ice_servers = response.json()

                    # Sử dụng trong webrtc_streamer
                   
                    # Sử dụng aiortc
                    
                    webrtc_ctx = webrtc_streamer(
                    key="camera-stream",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": ice_servers},
                    video_processor_factory=ObjectDetectionTransformer,
                    
                   media_stream_constraints = {
                    "video": {
                        "width": {"min": 1280, "ideal": 1920, "max": 3840},
                        "height": {"min": 720, "ideal": 1080, "max": 2160},
                        "frameRate": {"min": 15, "ideal": 30, "max": 60}
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

# Add this new class for QR code detection
class QRCodeProcessor(VideoProcessorBase):
    def __init__(self):
        self.qr_detected = False
        self.qr_data = None
        self.last_detection_time = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for QR detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect QR codes
        decoded_objects = decode(img_rgb)
        
        current_time = datetime.now().timestamp()
        
        for obj in decoded_objects:
            # Draw bounding box around QR code
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            
            # Draw the bounding box
            n = len(points)
            for j in range(0, n):
                cv2.line(img, tuple(points[j]), tuple(points[(j+1) % n]), (0, 255, 0), 3)
            
            # Add text overlay
            cv2.putText(img, "QR Code Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Store QR data (with rate limiting to avoid spam)
            if current_time - self.last_detection_time > 2:  # 2 second cooldown
                self.qr_data = obj.data.decode('utf-8')
                self.qr_detected = True
                self.last_detection_time = current_time
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Update the scan_qr_code function
def scan_qr_code():
    """Enhanced QR code scanning with WebRTC support"""
    st.markdown("<h2 style='text-align: center;'>Quét mã QR CCCD</h2>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Ảnh", "📷 Camera WebRTC"])
    
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
        <h3>Quét QR Code qua Camera</h3>
        <p>Sử dụng WebRTC để quét QR code trực tiếp từ camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize QR processor in session state
        if 'qr_processor' not in st.session_state:
            st.session_state.qr_processor = QRCodeProcessor()
        
        # WebRTC configuration for QR scanning
        try:
            if AIORTC_AVAILABLE:
                # Get ICE servers
                response = requests.get(
                    "https://iewcom.metered.live/api/v1/turn/credentials",
                    params={"apiKey": "5b0cc93867e02c9b2e8ef46de385169008aa"}
                )
                ice_servers = response.json()
                
                # Create WebRTC streamer for QR scanning
                webrtc_ctx = webrtc_streamer(
                    key="qr-scanner",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": ice_servers},
                    video_processor_factory=QRCodeProcessor,
                    media_stream_constraints = {
    "video": {
        "width": {"min": 1280, "ideal": 1920, "max": 3840},
        "height": {"min": 720, "ideal": 1080, "max": 2160},
        "frameRate": {"min": 15, "ideal": 30, "max": 60}
    },
    "audio": False
},

                    async_processing=False,
                    sendback_audio=False,
                )
                
                # Display instructions
                st.markdown("""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h4>📋 Hướng dẫn sử dụng:</h4>
                <ul>
                    <li>Nhấn "START" để bắt đầu camera</li>
                    <li>Đưa QR code vào khung hình</li>
                    <li>Hệ thống sẽ tự động phát hiện và xử lý QR code</li>
                    <li>QR code được phát hiện sẽ có khung màu xanh</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Check for QR detection
                if webrtc_ctx.video_processor:
                    processor = webrtc_ctx.video_processor
                    
                    # Display connection status
                    if webrtc_ctx.state.playing:
                        st.success("✅ Camera đang hoạt động - Sẵn sàng quét QR code")
                        
                        # Check if QR code was detected
                        if hasattr(processor, 'qr_detected') and processor.qr_detected:
                            st.balloons()
                            st.success("🎉 QR Code đã được phát hiện!")
                            
                            # Process the detected QR code
                            if hasattr(processor, 'qr_data') and processor.qr_data:
                                success, message = process_qr_detection(processor.qr_data)
                                if success:
                                    st.success(f"✅ {message}")
                                    # Display the processed citizen info
                                    display_latest_citizen_info()
                                else:
                                    st.error(f"❌ {message}")
                                
                                # Reset detection flag
                                processor.qr_detected = False
                                processor.qr_data = None
                    else:
                        st.info("📷 Nhấn 'START' để bắt đầu quét QR code")
                
            else:
                st.error("❌ WebRTC không khả dụng. Vui lòng sử dụng tab 'Upload Ảnh'")
                
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo camera: {str(e)}")
            st.info("💡 Thử làm mới trang hoặc sử dụng tab 'Upload Ảnh'")
            
            # Fallback to simple camera input
            st.markdown("---")
            st.markdown("### 📷 Camera đơn giản (Fallback)")
            camera_image = st.camera_input("Chụp ảnh QR Code", key="qr_camera_fallback")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                st.image(image, caption="Ảnh đã chụp", use_container_width=True)
                
                if st.button("Xử lý QR Code", key="process_camera_qr_fallback"):
                    with st.spinner("Đang xử lý..."):
                        success, message = process_image_for_qr(image)
                        if success:
                            st.success(message)
                        else:
                            st.error("Không tìm thấy mã QR trong ảnh. Vui lòng thử lại.")
def display_latest_citizen_info():
    """Display information of the most recently added citizen"""
    if not st.session_state.citizens_data.empty:
        latest_citizen = st.session_state.citizens_data.iloc[-1]
        
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h4 style="color: #2e7d32;">📋 Thông tin công dân vừa quét:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {latest_citizen['id']}")
            st.write(f"**Số CCCD:** {latest_citizen['cccd']}")
            st.write(f"**Họ và tên:** {latest_citizen['name']}")
            st.write(f"**Ngày sinh:** {latest_citizen['dob']}")
        
        with col2:
            st.write(f"**Giới tính:** {latest_citizen['sex']}")
            st.write(f"**Địa chỉ:** {latest_citizen['address']}")
            st.write(f"**Ngày hết hạn:** {latest_citizen['expdate']}")
            st.write(f"**Thời gian quét:** {latest_citizen['scan_date']}")

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