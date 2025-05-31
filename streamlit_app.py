import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

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

# Danh su00e1ch tu00e0i khou1ea3n mu1eabu (trong thu1ef1c tu1ebf nu00ean lu01b0u trong cu01a1 su1edf du1eef liu1ec7u vu00e0 mu00e3 hu00f3a mu1eadt khu1ea9u)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

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
                st.experimental_rerun()
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
        
        # Camera stream vu1edbi object detection
        webrtc_ctx = webrtc_streamer(
            key="surveillance",
            video_transformer_factory=ObjectDetectionTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )


    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Camera Giám Sát</h3>
            <p>Giám sát trực tiếp từ camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cu00e1c nu00fat u0111iu1ec1u khiu1ec3n
        if st.button("Bắt đầu giám sát"):
            st.session_state.surveillance_active = True
         #   st.experimental_rerun()
            st.experimental_rerun()
            
        detection_options = st.multiselect(
            "Chọn các đối tượng cần phát hiện:",
            ["Khuôn mặt", "Phương tiện", "Vật thể khả nghi"],
            default=["Khuôn mặt"]
        )
        
        sensitivity = st.slider("Độ nhạy phát hiện", 0, 100, 50)
        
        if st.button("Chụp ảnh"):
            st.success("Ảnh đã được chụp thành công!")

# Thu00eam class cho object detection
class ObjectDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Thu00eam logic phu00e1t hiu1ec7n u0111u1ed1i tu01b0u1ee3ng u1edf u0111u00e2y
        # (Cu00f3 thu1ec3 su1eed du1ee5ng OpenCV, YOLO, hou1eb7c cu00e1c model khu00e1c)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def init_camera():
    """
    Khu1edfi tu1ea1o camera vu00e0 kiu1ec3m tra ku1ebft nu1ed1i
    """
    try:
        # Thu1eed ku1ebft nu1ed1i vu1edbi camera cu1ee7a thiu1ebft bu1ecb
        camera = cv2.VideoCapture(0)
        
        # Kiu1ec3m tra xem camera cu00f3 hou1ea1t u0111u1ed9ng khu00f4ng
        if not camera.isOpened():
            st.error("Không thể kết nối với camera!")
            return None
            
        return camera
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None

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

# Thu00eam vu00e0o u0111u1ea7u file
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class QRCodeVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.qr_detected = False
        self.qr_data = None

    def transform(self, frame):
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
    """
    Chu1ee9c nu0103ng quu00e9t mu00e3 QR tu1eeb camera hou1eb7c u1ea3nh tu1ea3i lu00ean
    """
    st.markdown("<h2 style='text-align: center;'>Quét mã QR CCCD</h2>", unsafe_allow_html=True)
    
    # Chia layout thu00e0nh 2 cu1ed9t
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Hướng dẫn</h3>
            <p>Định dạng hỗ trợ: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Tải lên ảnh", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
            
            if st.button("Tải ảnh lên"):
                success, message = process_image_for_qr(image)
                if success:
                    st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Qúet qua Camera</h3>
            <p>Vui lòng đảm bảo rằng camera của bạn đã được kết nối và hoạt động bình thường.</p>
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4 style="color: #856404;">Hướng dẫn sử dụng:</h4>
                <ol style="color: #856404;">
                    <li>Kiểm tra kết nối và cấp quyền truy cập camera trước khi sử dụng</li>
                    <li>Đặt camera ở vị trí phù hợp để quét mã QR một cách chính xác, ấn nút Allow</li>
                    <li>Đảm bảo ánh sáng đủ để camera có thể nhận diện mã QR</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Khu1edfi tu1ea1o WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="qr-scanner",
            video_transformer_factory=QRCodeVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

       # Trong hu00e0m transform cu1ee7a class QRCodeVideoTransformer, su1eeda phu1ea7n xu1eed lu00fd khi phu00e1t hiu1ec7n QR:
        if webrtc_ctx.video_transformer:
            if webrtc_ctx.video_transformer.qr_detected:
                qr_data = webrtc_ctx.video_transformer.qr_data
                citizen_info = qr_data.split('|')
                
                if len(citizen_info) >= 7:
                    st.success("QR code detected and processed successfully!")
                    
                    # Lu01b0u thu00f4ng tin vu00e0o DataFrame
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
                    
                    # Hiu1ec3n thu1ecb thu00f4ng tin
                    st.markdown("""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="color: #2e7d32;">Thông tin công dân:</h4>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**ID:** {citizen_info[0]}")
                    st.write(f"**Số CCCD:** {citizen_info[1]}")
                    st.write(f"**Họ và tên:** {citizen_info[2]}")
                    st.write(f"**Ngày sinh:** {citizen_info[3]}")
                    st.write(f"**Giới tính:** {citizen_info[4]}")
                    st.write(f"**Địa chỉ:** {citizen_info[5]}")
                    st.write(f"**Ngày hết hạn:** {citizen_info[6]}")


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
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng quu00e9t QR
        if st.button("Quét QR CCCD"):
            st.session_state.page = "scan_qr"
            scan_qr_code()
          #  st.experimental_rerun()
        
    with col2:
        st.markdown("""
        <div class="feature-button">
            <h3>Quản lý Công dân</h3>
            <p>Quản lý dữ liệu công dân hiệu quả và dễ dàng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nu00fat ku1ebft nu1ed1i vu1edbi chu1ee9c nu0103ng quu1ea3n lu00fd du1eef liu1ec7u
        if st.button("Xem dữ liệu công dân"):
            st.session_state.page = "view_data"
            show_citizen_data()
          #  st.experimental_rerun()
        
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
            surveillance_camera()
           # st.experimental_rerun()
    
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
    st.sidebar.markdown("<h1 style='text-align: center;'>ud83cudfdbufe0f</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center;'>Quản lý Công dân</h2>", unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb thu00f4ng tin ngu01b0u1eddi du00f9ng u0111u0103ng nhu1eadp
    st.sidebar.markdown(f"""<div style='text-align: center; padding: 10px; background-color: #e8f5e9; 
                        border-radius: 5px; margin-bottom: 20px;'>
                        ud83dudc64 <b>{st.session_state.username}</b></div>""", 
                        unsafe_allow_html=True)
    
    menu = [
        "Trang chủ",
        "Quét QR CCCD",
        "Xem dữ liệu",
        "Camera Giám sát",
        "Thống kê",
        "Cài đặt"
    ]
    
    choice = st.sidebar.selectbox("Chọn chức năng", menu)
    
    # Hiu1ec3n thu1ecb cu00e1c nu00fat chu1ee9c nu0103ng phu1ee5 trong sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Chức năng nhanh")
    
    if st.sidebar.button("📷 Camera"):
        st.session_state.page = "camera"
        surveillance_camera()
      #  st.experimental_rerun()
        
    if st.sidebar.button("📊 Báo cáo"):
        st.session_state.page = "reports"
        show_statistics()
       # st.experimental_rerun()
        
    if st.sidebar.button("⚙️ Cài đặt"):
        st.session_state.page = "settings"
        show_settings()
      #  st.experimental_rerun()
    
    # Nu00fat u0111u0103ng xuu1ea5t
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()
    
    # Xu1eed lu00fd cu00e1c trang
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