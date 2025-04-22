import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image

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
    st.session_state.citizens_data = pd.DataFrame(columns=['id', 'name', 'dob', 'address', 'scan_date', 'image_path'])

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
            
            if len(citizen_info) >= 4:
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
                    'name': citizen_info[1],
                    'dob': citizen_info[2],
                    'address': citizen_info[3],
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
                    <li>Khi bấm "Bật Camera", trình duyệt sẽ yêu cầu quyền truy cập camera</li>
                    <li>Vui lòng chọn "Allow" hoặc "Cho phép" để sử dụng tính năng này</li>
                    <li>Nếu đã từ chối trước đó, vui lòng:
                        <ul>
                            <li>Kiểm tra biểu tượng camera trên thanh địa chỉ</li>
                            <li>Hoặc vào cài đặt trình duyệt để cấp quyền camera</li>
                        </ul>
                    </li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        start_camera = st.button("Bật Camera")
        
        if start_camera:
            # Hiển thị thông báo đang khởi tạo camera
            with st.spinner('Đang kết nối với camera...'):
                camera = init_camera()
                
            if camera is not None:
                st.success("Đã kết nối camera thành công!")
                frame_placeholder = st.empty()
                stop_button = st.button("Dừng quét")
                
                try:
                    while not stop_button:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("Không thể đọc frame từ camera")
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        success, message = process_image_for_qr(frame_rgb)
                        
                        if success:
                            st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
                            break
                        
                        frame_placeholder.image(frame_rgb, channels="RGB")
                except Exception as e:
                    st.error(f"""
                    Lỗi khi sử dụng camera: {str(e)}
                    
                    Vui lòng kiểm tra:
                    1. Quyền truy cập camera trong trình duyệt
                    2. Camera có đang được ứng dụng khác sử dụng không
                    3. Camera có được kết nối đúng cách không
                    """)
                finally:
                    camera.release()
            else:
                st.error("""
                Không thể kết nối với camera!
                
                Vui lòng kiểm tra:
                1. Camera có được kết nối với máy tính không
                2. Quyền truy cập camera trong trình duyệt
                3. Camera không bị ứng dụng khác sử dụng
                """)


def show_citizen_data():
    """
    Hiển thị dữ liệu công dân đã quét
    """
    st.markdown("<h2 style='text-align: center;'>Dữ liệu Công dân</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Công dân: {row['name']} - ID: {row['id']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="Ảnh CCCD", use_column_width=True)
                    else:
                        st.warning("Ảnh không khả dụng")
                
                with col2:
                    st.markdown(f"""
                    **ID:** {row['id']}  
                    **Họ tên:** {row['name']}  
                    **Ngày sinh:** {row['dob']}  
                    **Địa chỉ:** {row['address']}  
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
