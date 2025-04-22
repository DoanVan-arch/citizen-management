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

# Thiết lập giao diện trang
st.set_page_config(
    page_title="Hệ thống Quản lý Công dân",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    }
    .success-message {
        padding: 20px;
        background-color: #4CAF50;
        color: white;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .error-message {
        padding: 20px;
        background-color: #f44336;
        color: white;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .info-card {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo session state
if 'citizens_data' not in st.session_state:
    st.session_state.citizens_data = pd.DataFrame(columns=[
        'id', 'cccd', 'name', 'dob', 'sex', 'address', 'expdate', 'scan_date', 'image_path'
    ])


def init_camera():
    """
    Khởi tạo camera và kiểm tra kết nối
    """
    try:
        # Thử kết nối với camera của thiết bị
        camera = cv2.VideoCapture(0)
        
        # Kiểm tra xem camera có hoạt động không
        if not camera.isOpened():
            st.error("Không thể kết nối với camera. Vui lòng kiểm tra lại thiết bị.")
            return None
            
        return camera
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo camera: {str(e)}")
        return None

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
                
                return True, "Quét QR thành công!"
                
        return False, "Không tìm thấy mã QR trong ảnh."
    
    except Exception as e:
        return False, f"Lỗi khi xử lý ảnh: {str(e)}"

# Thêm vào đầu file
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class QRCodeVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.qr_detected = False
        self.qr_data = None

    def transform(self, frame):
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

def scan_qr_code():
    """
    Chức năng quét mã QR từ camera hoặc ảnh tải lên
    """
    st.markdown("<h2 style='text-align: center;'>Quét mã QR CCCD</h2>", unsafe_allow_html=True)
    
    # Chia layout thành 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Tải lên ảnh CCCD</h3>
            <p>Hỗ trợ các định dạng: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Chọn ảnh CCCD", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
            
            if st.button("Xử lý ảnh"):
                success, message = process_image_for_qr(image)
                if success:
                    st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Quét qua Camera</h3>
            <p>Sử dụng camera để quét trực tiếp</p>
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4 style="color: #856404;">⚠️ Lưu ý quan trọng:</h4>
                <ol style="color: #856404;">
                    <li>Khi bấm "START", trình duyệt sẽ yêu cầu quyền truy cập camera</li>
                    <li>Vui lòng chọn "Allow" hoặc "Cho phép" để sử dụng tính năng này</li>
                    <li>Đảm bảo camera không bị ứng dụng khác sử dụng</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Khởi tạo WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="qr-scanner",
            video_transformer_factory=QRCodeVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

       # Trong hàm transform của class QRCodeVideoTransformer, sửa phần xử lý khi phát hiện QR:
        if webrtc_ctx.video_transformer:
            if webrtc_ctx.video_transformer.qr_detected:
                qr_data = webrtc_ctx.video_transformer.qr_data
                citizen_info = qr_data.split('|')
                
                if len(citizen_info) >= 7:
                    st.success("Đã quét thành công QR Code!")
                    
                    # Lưu thông tin vào DataFrame
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
                    
                    # Hiển thị thông tin
                    st.markdown("""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="color: #2e7d32;">Thông tin công dân:</h4>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**ID:** {citizen_info[0]}")
                    st.write(f"**Số CCCD:** {citizen_info[1]}")
                    st.write(f"**Họ tên:** {citizen_info[2]}")
                    st.write(f"**Ngày sinh:** {citizen_info[3]}")
                    st.write(f"**Giới tính:** {citizen_info[4]}")
                    st.write(f"**Địa chỉ:** {citizen_info[5]}")
                    st.write(f"**Ngày hết hạn:** {citizen_info[6]}")


def show_citizen_data():
    st.markdown("<h2 style='text-align: center;'>Dữ liệu Công dân</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Công dân: {row['name']} - CCCD: {row['cccd']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="Ảnh CCCD", use_column_width=True)
                    else:
                        st.warning("Ảnh không khả dụng")
                
                with col2:
                    st.markdown(f"""
                    **ID:** {row['id']}  
                    **Số CCCD:** {row['cccd']}  
                    **Họ tên:** {row['name']}  
                    **Ngày sinh:** {row['dob']}  
                    **Giới tính:** {row['sex']}  
                    **Địa chỉ:** {row['address']}  
                    **Ngày hết hạn:** {row['expdate']}  
                    **Ngày quét:** {row['scan_date']}
                    """)
    else:
        st.info("Chưa có dữ liệu công dân nào.")


def main():
    """
    Hàm chính của ứng dụng
    """
    # Sidebar
    st.sidebar.markdown("<h1 style='text-align: center;'>🏛️</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center;'>Quản lý Công dân</h2>", unsafe_allow_html=True)
    
    menu = ["Trang chủ", "Quét QR CCCD", "Xem dữ liệu"]
    choice = st.sidebar.selectbox("Chọn chức năng", menu)
    
    if choice == "Trang chủ":
        st.markdown("<h1 style='text-align: center;'>Hệ thống Quản lý Công dân</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h2>Chào mừng! 👋</h2>
            <p>Đây là hệ thống quản lý thông tin công dân thông qua quét mã QR trên CCCD.</p>
            <h3>Các chức năng chính:</h3>
            <ul>
                <li>Quét QR từ CCCD qua camera</li>
                <li>Tải lên ảnh CCCD để quét</li>
                <li>Xem và quản lý dữ liệu công dân</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif choice == "Quét QR CCCD":
        scan_qr_code()
        
    elif choice == "Xem dữ liệu":
        show_citizen_data()

if __name__ == '__main__':
    main()
