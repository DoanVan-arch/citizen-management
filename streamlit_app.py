from operator import truediv
import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image
#from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration 
import av
from contextlib import contextmanager  # Add this import
# d:\Codes\citizen-management\streamlit_app.py


# Thêm try-except cho import asyncio để xử lý lỗi liên quan đến asyncio
try:
    import asyncio
    import threading
except ImportError:
    st.error("Không thể import asyncio. Một số chức năng có thể không hoạt động.")

# Cu1eadp nhu1eadt import cho streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    WEBRTC_AVAILABLE = False
    st.warning(f"streamlit-webrtc khu00f4ng khu1ea3 du1ee5ng: {str(e)}. Chu1ec9 su1eed du1ee5ng chu1ee9c nu0103ng upload u1ea3nh.")
# Thiu1ebft lu1eadp giao diu1ec7n trang
st.set_page_config(
    page_title="HỆ THỐNG QUẢN LÝ CÔNG DÂN",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tu00f9y chu1ec9nh
# Cu1eadp nhu1eadt phu1ea7n CSS
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
    
# Thêm session state cho menu choice
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Trang chủ"

# Danh su00e1ch tu00e0i khou1ea3n mu1eabu (trong thu1ef1c tu1ebf nu00ean lu01b0u trong cu01a1 su1edf du1eef liu1ec7u vu00e0 mu00e3 hu00f3a mu1eadt khu1ea9u)
USERS = {
    "admin": "admin123",
    "user": "user123"
}
def setup_asyncio():
    """Setup asyncio event loop for WebRTC"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
@contextmanager
def webrtc_context():
    """Context manager for WebRTC operations"""
    try:
        setup_asyncio()
        yield
    except Exception as e:
        st.error(f"WebRTC Error: {str(e)}")
        st.info("Please try refreshing the page or use the image upload feature instead.")

def safe_webrtc_streamer(**kwargs):
    """Enhanced wrapper for webrtc_streamer with better error handling"""
    try:
        if not WEBRTC_AVAILABLE:
            st.error("WebRTC không khả dụng. Vui lòng sử dụng chức năng upload ảnh.")
            return None
        
        # Setup asyncio properly
        setup_asyncio()
        
        # Remove problematic parameters and set safe defaults
        safe_kwargs = kwargs.copy()
        safe_kwargs['async_processing'] = False
        
        # Add error handling for media constraints
        if 'media_stream_constraints' not in safe_kwargs:
            safe_kwargs['media_stream_constraints'] = {
                "video": {"width": 640, "height": 480, "frameRate": 15}, 
                "audio": False
            }
        
        with webrtc_context():
            return webrtc_streamer(**safe_kwargs)
            
    except Exception as e:
        st.error(f"Lỗi WebRTC: {str(e)}")
        st.info("Hãy thử làm mới trang hoặc sử dụng chức năng upload ảnh thay thế.")
        return None

# Update RTC Configuration with more reliable settings
if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}

           
        ],
        "iceTransportPolicy": "all",
        "bundlePolicy": "balanced"
    })
else:
    RTC_CONFIGURATION = None

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
        
        # Add option to choose between WebRTC and file upload
        camera_option = st.radio(
            "Chọn phương thức:",
            ["Camera trực tiếp (WebRTC)", "Upload video/ảnh"],
            key="camera_option"
        )
        
        if camera_option == "Camera trực tiếp (WebRTC)":
            try:
                # Enhanced WebRTC streamer with better error handling
                webrtc_ctx = safe_webrtc_streamer(
                    key="surveillance",
                   # video_processor_factory=ObjectDetectionTransformer,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={
                        "video": {"width": 320, "height": 240, "frameRate": 15},
                        "audio": False
                    },
                    async_processing=False,
                )
                
                # Display connection status
                if webrtc_ctx and webrtc_ctx.state.playing:
                    st.success("✅ Camera đang hoạt động")
                elif webrtc_ctx and webrtc_ctx.state.signalling:
                    st.warning("🔄 Đang kết nối camera...")
                else:
                    st.info("📷 Nhấn 'START' để bắt đầu camera")
                    
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
                    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                    
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
                
        return False, "Lỗi không xác định."
    
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

# Cấu hình RTC với nhiều STUN servers để tăng độ tin cậy
class ObjectDetectionTransformer(VideoProcessorBase):
    def __init__(self):
        self.qr_detected = False
        self.qr_data = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class QRCodeVideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.qr_detected = False
        self.qr_data = None

    def recv(self, frame):
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
            st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
            
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
        
        # Add option to choose scanning method
        scan_method = st.radio(
            "Chọn phương thức quét:",
            ["Camera WebRTC", "Chụp ảnh từ camera"],
            key="scan_method"
        )
        
        if scan_method == "Camera WebRTC":
            try:
                webrtc_ctx = safe_webrtc_streamer(
                    key="qr-scanner",
                   # video_processor_factory=QRCodeVideoTransformer,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={
                        "video": {"width": 640, "height": 480, "frameRate": 15},
                        "audio": False
                    },
                    async_processing=False,
                )
                
                # Process QR detection results
                if webrtc_ctx and webrtc_ctx.video_transformer:
                    if hasattr(webrtc_ctx.video_transformer, 'qr_detected') and webrtc_ctx.video_transformer.qr_detected:
                        process_qr_detection(webrtc_ctx.video_transformer.qr_data)
                        
            except Exception as e:
                st.error(f"Lỗi camera: {str(e)}")
                st.info("Vui lòng thử sử dụng tùy chọn 'Chụp ảnh từ camera' hoặc 'Upload Ảnh'")
        
        else:
            # Alternative camera capture method
            if st.button("Chụp ảnh từ camera", key="capture_camera"):
                st.info("Chức năng này đang được phát triển. Vui lòng sử dụng tùy chọn upload ảnh.")

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
    if 'webrtc_error_count' not in st.session_state:
        st.session_state.webrtc_error_count = 0
    
    if st.session_state.webrtc_error_count > 3:
        st.warning("Phát hiện nhiều lỗi WebRTC. Đang reset session...")
        for key in list(st.session_state.keys()):
            if 'webrtc' in key.lower():
                del st.session_state[key]
        st.session_state.webrtc_error_count = 0
        st.rerun()
def show_citizen_data():
    st.markdown("<h2 style='text-align: center;'>Dữ liệu Công dân</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Công dân:{row['name']} - CCCD: {row['cccd']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="ảnh CCCD", use_column_width=True)
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
    
    # Grid layout cho cu00e1c chu1ee9c nu0103ng
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
            #show_citizen_data()
          #  st.rerun()
        
    with col3:
        st.markdown("""
        <div class="feature-button">
            <h3>Camera Giám Sát</h3>
            <p>Theo dõi qua camera an ninh</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng camera giu00e1m su00e1t
        if st.button("Camera giám sát"):
            st.session_state.page = "camera"
            st.session_state.menu_choice = "Camera Giám sát"
           # surveillance_camera()
         #   st.rerun()
    
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
    st.markdown("<h2 style='text-align: center;'>Thống kê</h2>", unsafe_allow_html=True)
    st.write("Hiển thị các số liệu thống kê liên quan đến công dân.")
    # Thu00eam code hiu1ec3n thu1ecb thu1ed1ng ku00ea

def show_settings():
    st.markdown("<h2 style='text-align: center;'>Cài đặt</h2>", unsafe_allow_html=True)
    st.write("Tùy chỉnh các thiết lập của hệ thống tại đây.")
    # Thu00eam code cu00e0i u0111u1eb7t


def main():
    # Kiu1ec3m tra u0111u0103ng nhu1eadp
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Hiu1ec3n thu1ecb giao diu1ec7n chu00ednh sau khi u0111u0103ng nhu1eadp
    st.sidebar.markdown("<h1 style='text-align: center;'>Chào mừng 📷</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center;'>Quản lý Công dân</h2>", unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb thu00f4ng tin ngu01b0u1eddi du00f9ng u0111u0103ng nhu1eadp
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
    
    # Hiu1ec3n thu1ecb cu00e1c nu00fat chu1ee9c nu0103ng phu1ee5 trong sidebar
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
        #show_settings()
        st.session_state.menu_choice = "Cài đặt"
       # st.rerun()
    
    # Nu00fat u0111u0103ng xuu1ea5t
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
    # Xu1eed lu00fd cu00e1c trang
    


if __name__ == '__main__':
    main()