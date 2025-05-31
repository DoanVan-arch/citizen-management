import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import tempfile

# Thiết lập giao diện trang
st.set_page_config(
    page_title="HỆ THỐNG QUẢN LÝ CÔNG DÂN",
    page_icon="📋",
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

# Thêm session state cho QR data
if 'qr_data' not in st.session_state:
    st.session_state.qr_data = None

# Danh sách tài khoản mẫu (trong thực tế nên lưu trong cơ sở dữ liệu và mã hóa mật khẩu)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

# Hàm xử lý ảnh để phát hiện QR code
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
    
    return image, decoded_objects

# Hàm xử lý ảnh đã tải lên
def process_uploaded_image(uploaded_file):
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Process for QR code
    processed_image, decoded_objects = process_image_for_qr(image)
    
    return processed_image, decoded_objects

# Hàm đăng nhập
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

# Hàm xử lý video frame
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    flipped = img[::-1,:,:]
    return flipped

# Hàm giám sát camera
def surveillance_camera():
    st.markdown("<h1 style='text-align: center;'>Giám sát an ninh</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Giám sát an ninh</h3>
        <p>Theo dõi và phát hiện đối tượng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose between camera and file upload
        camera_option = st.radio(
            "Chọn phương thức:",
            ["Camera trực tiếp", "Upload video/ảnh"],
            key="camera_option"
        )
        
        if camera_option == "Camera trực tiếp":
            try:
                # Use Streamlit's native camera input
                camera_image = st.camera_input("Camera giám sát", key="surveillance_camera")
                
                if camera_image is not None:
                    # Process the captured image
                    image = Image.open(camera_image)
                    image_array = np.array(image)
                    
                    # Flip the image horizontally for a mirror effect
                    flipped_image = cv2.flip(image_array, 1)
                    
                    # Display the processed image
                    st.image(flipped_image, caption="Camera Feed", use_container_width=True)
                    
                    # Save button
                    if st.button("Lưu ảnh", key="save_surveillance"):
                        # Create temp directory if it doesn't exist
                        if not os.path.exists("captured_images"):
                            os.makedirs("captured_images")
                        
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_images/surveillance_{timestamp}.jpg"
                        Image.fromarray(flipped_image).save(filename)
                        st.success(f"Ảnh đã được lưu tại {filename}")
                        
            except Exception as e:
                st.error(f"Lỗi kết nối camera: {str(e)}")
                st.info("Vui lòng thử sử dụng tùy chọn 'Upload video/ảnh' bên dưới")
                
        else:
            # Alternative: File upload for surveillance
            uploaded_file = st.file_uploader("Tải lên video hoặc ảnh", type=["jpg", "jpeg", "png", "mp4", "avi"])
            
            if uploaded_file is not None:
                # Check if it's an image or video
                file_type = uploaded_file.type
                
                if "image" in file_type:
                    # Process image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
                    
                elif "video" in file_type:
                    # Save video to temp file and process
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    
                    # Display video
                    st.video(tfile.name)
                    
                    # Clean up temp file
                    os.unlink(tfile.name)
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Thông tin giám sát</h3>
        <p>Dữ liệu và cảnh báo từ hệ thống giám sát</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display some mock surveillance data
        st.markdown("### Trạng thái hệ thống")
        st.success("✅ Hệ thống đang hoạt động bình thường")
        
        st.markdown("### Thống kê")
        col_a, col_b = st.columns(2)
        col_a.metric("Đối tượng phát hiện", "0")
        col_b.metric("Cảnh báo", "0")
        
        st.markdown("### Nhật ký hoạt động")
        st.text("10:30:45 - Khởi động hệ thống")
        st.text("10:31:12 - Kết nối camera thành công")
        st.text(f"{datetime.now().strftime('%H:%M:%S')} - Đang giám sát...")

# Hàm quét mã QR
def scan_qr_code():
    st.markdown("<h1 style='text-align: center;'>Quét mã QR</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Quét mã QR</h3>
        <p>Quét mã QR để truy xuất thông tin công dân</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose between camera and file upload
        qr_option = st.radio(
            "Chọn phương thức:",
            ["Camera trực tiếp", "Upload ảnh"],
            key="qr_option"
        )
        
        if qr_option == "Camera trực tiếp":
            try:
                # Use Streamlit's native camera input
                qr_camera_image = st.camera_input("Quét mã QR", key="qr_camera")
                
                if qr_camera_image is not None:
                    # Process the captured image
                    image = Image.open(qr_camera_image)
                    image_array = np.array(image)
                    
                    # Process for QR code
                    processed_image, decoded_objects = process_image_for_qr(image_array)
                    
                    # Display the processed image
                    st.image(processed_image, caption="Processed Image", use_container_width=True)
                    
                    # Display QR data if detected
                    if len(decoded_objects) > 0:
                        for obj in decoded_objects:
                            qr_data = obj.data.decode('utf-8')
                            st.success(f"Đã phát hiện mã QR: {qr_data}")
                            
                            # Parse QR data (assuming it's in JSON or key-value format)
                            try:
                                # Try to parse as JSON
                                import json
                                qr_json = json.loads(qr_data)
                                st.json(qr_json)
                                
                                # Add to citizens data if it contains required fields
                                if all(k in qr_json for k in ['cccd', 'name']):
                                    new_citizen = {
                                        'id': len(st.session_state.citizens_data) + 1,
                                        'cccd': qr_json.get('cccd', ''),
                                        'name': qr_json.get('name', ''),
                                        'dob': qr_json.get('dob', ''),
                                        'sex': qr_json.get('sex', ''),
                                        'address': qr_json.get('address', ''),
                                        'expdate': qr_json.get('expdate', ''),
                                        'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'image_path': None
                                    }
                                    
                                    # Save image if needed
                                    if not os.path.exists("citizen_images"):
                                        os.makedirs("citizen_images")
                                    
                                    image_path = f"citizen_images/citizen_{new_citizen['cccd']}.jpg"
                                    Image.fromarray(processed_image).save(image_path)
                                    new_citizen['image_path'] = image_path
                                    
                                    # Add to dataframe
                                    st.session_state.citizens_data = pd.concat([
                                        st.session_state.citizens_data, 
                                        pd.DataFrame([new_citizen])
                                    ], ignore_index=True)
                                    
                                    st.success(f"Đã thêm công dân {new_citizen['name']} vào hệ thống!")
                            except:
                                # If not JSON, just display as text
                                st.text(qr_data)
                    else:
                        st.info("Không phát hiện mã QR. Vui lòng thử lại.")
                        
            except Exception as e:
                st.error(f"Lỗi xử lý QR: {str(e)}")
                
        else:
            # Alternative: File upload for QR scanning
            uploaded_file = st.file_uploader("Tải lên ảnh có chứa mã QR", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Process uploaded image for QR
                processed_image, decoded_objects = process_uploaded_image(uploaded_file)
                st.image(processed_image, caption="Ảnh đã xử lý", use_container_width=True)
                
                # Display QR data if detected
                if len(decoded_objects) > 0:
                    for obj in decoded_objects:
                        qr_data = obj.data.decode('utf-8')
                        st.success(f"Đã phát hiện mã QR: {qr_data}")
                        
                        # Try to parse as JSON
                        try:
                            import json
                            qr_json = json.loads(qr_data)
                            st.json(qr_json)
                            
                            # Add to citizens data if it contains required fields
                            if all(k in qr_json for k in ['cccd', 'name']):
                                new_citizen = {
                                    'id': len(st.session_state.citizens_data) + 1,
                                    'cccd': qr_json.get('cccd', ''),
                                    'name': qr_json.get('name', ''),
                                    'dob': qr_json.get('dob', ''),
                                    'sex': qr_json.get('sex', ''),
                                    'address': qr_json.get('address', ''),
                                    'expdate': qr_json.get('expdate', ''),
                                    'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'image_path': None
                                }
                                
                                # Save image if needed
                                if not os.path.exists("citizen_images"):
                                    os.makedirs("citizen_images")
                                
                                image_path = f"citizen_images/citizen_{new_citizen['cccd']}.jpg"
                                Image.fromarray(processed_image).save(image_path)
                                new_citizen['image_path'] = image_path
                                
                                # Add to dataframe
                                st.session_state.citizens_data = pd.concat([
                                    st.session_state.citizens_data, 
                                    pd.DataFrame([new_citizen])
                                ], ignore_index=True)
                                
                                st.success(f"Đã thêm công dân {new_citizen['name']} vào hệ thống!")
                        except:
                            # If not JSON, just display as text
                            st.text(qr_data)
                else:
                    st.warning("Không phát hiện mã QR trong ảnh đã tải lên.")
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Hướng dẫn</h3>
        <p>Cách quét mã QR để truy xuất thông tin</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        1. **Sử dụng camera**: Đặt mã QR vào khung hình và giữ yên
        2. **Upload ảnh**: Tải lên ảnh có chứa mã QR
        3. **Xử lý dữ liệu**: Hệ thống sẽ tự động phát hiện và xử lý mã QR
        4. **Lưu thông tin**: Dữ liệu sẽ được lưu vào hệ thống nếu hợp lệ
        """)
        
        st.markdown("### Mẫu dữ liệu QR hợp lệ")
        st.code('''
        {
            "cccd": "079202012345",
            "name": "Nguyễn Văn A",
            "dob": "01/01/1990",
            "sex": "Nam",
            "address": "123 Đường ABC, Quận XYZ, TP.HCM",
            "expdate": "01/01/2030"
        }
        ''')

# Hàm quản lý dữ liệu công dân
def manage_citizens():
    st.markdown("<h1 style='text-align: center;'>Quản lý dữ liệu công dân</h1>", unsafe_allow_html=True)
    
    # Hiển thị dữ liệu công dân
    if len(st.session_state.citizens_data) > 0:
        st.markdown("""
        <div class="info-card">
        <h3>Danh sách công dân</h3>
        <p>Thông tin công dân đã được lưu trong hệ thống</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị bảng dữ liệu
        st.dataframe(st.session_state.citizens_data[['id', 'cccd', 'name', 'dob', 'sex', 'address', 'scan_date']])
        
        # Chọn công dân để xem chi tiết
        citizen_ids = st.session_state.citizens_data['id'].tolist()
        selected_id = st.selectbox("Chọn ID công dân để xem chi tiết", citizen_ids)
        
        if selected_id:
            # Lấy thông tin công dân
            citizen = st.session_state.citizens_data[st.session_state.citizens_data['id'] == selected_id].iloc[0]
            
            st.markdown("### Thông tin chi tiết công dân")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if citizen['image_path'] and os.path.exists(citizen['image_path']):
                    st.image(citizen['image_path'], caption="ảnh CCCD", use_container_width=True)
                else:
                    st.info("Không có ảnh")
            
            with col2:
                st.markdown(f"**Họ tên:** {citizen['name']}")
                st.markdown(f"**CCCD:** {citizen['cccd']}")
                st.markdown(f"**Ngày sinh:** {citizen['dob']}")
                st.markdown(f"**Giới tính:** {citizen['sex']}")
                st.markdown(f"**Địa chỉ:** {citizen['address']}")
                st.markdown(f"**Ngày hết hạn:** {citizen['expdate']}")
                st.markdown(f"**Ngày quét:** {citizen['scan_date']}")
                
                # Nút xóa công dân
                if st.button("Xóa công dân này"):
                    st.session_state.citizens_data = st.session_state.citizens_data[st.session_state.citizens_data['id'] != selected_id]
                    st.success("Đã xóa thông tin công dân!")
                    st.rerun()
    else:
        st.info("Chưa có dữ liệu công dân nào. Vui lòng quét mã QR để thêm công dân.")
    
    # Thêm công dân mới thủ công
    st.markdown("""
    <div class="info-card">
    <h3>Thêm công dân mới</h3>
    <p>Nhập thông tin công dân mới vào hệ thống</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_cccd = st.text_input("Số CCCD")
        new_name = st.text_input("Họ tên")
        new_dob = st.date_input("Ngày sinh")
        new_sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
    
    with col2:
        new_address = st.text_area("Địa chỉ")
        new_expdate = st.date_input("Ngày hết hạn")
        new_image = st.file_uploader("Tải lên ảnh CCCD", type=["jpg", "jpeg", "png"])
    
    if st.button("Thêm công dân"):
        if new_cccd and new_name:
            # Tạo ID mới
            new_id = len(st.session_state.citizens_data) + 1
            
            # Xử lý ảnh nếu có
            image_path = None
            if new_image:
                if not os.path.exists("citizen_images"):
                    os.makedirs("citizen_images")
                
                image_path = f"citizen_images/citizen_{new_cccd}.jpg"
                Image.open(new_image).save(image_path)
            
            # Tạo dữ liệu công dân mới
            new_citizen = {
                'id': new_id,
                'cccd': new_cccd,
                'name': new_name,
                'dob': new_dob.strftime("%d/%m/%Y"),
                'sex': new_sex,
                'address': new_address,
                'expdate': new_expdate.strftime("%d/%m/%Y"),
                'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_path': image_path
            }
            
            # Thêm vào dataframe
            st.session_state.citizens_data = pd.concat([
                st.session_state.citizens_data, 
                pd.DataFrame([new_citizen])
            ], ignore_index=True)
            
            st.success(f"Đã thêm công dân {new_name} vào hệ thống!")
            st.rerun()
        else:
            st.error("Vui lòng nhập đầy đủ thông tin bắt buộc (CCCD và Họ tên)")

# Hàm trang chủ
def home_page():
    st.markdown("<h1 style='text-align: center;'>HỆ THỐNG QUẢN LÝ CÔNG DÂN</h1>", unsafe_allow_html=True)
    
    # Hiển thị thông tin người dùng
    st.markdown(f"""
    <div class="info-card">
    <h3>Xin chào, {st.session_state.username}!</h3>
    <p>Chào mừng bạn đến với Hệ thống Quản lý Công dân. Vui lòng chọn chức năng bên dưới để bắt đầu.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị các chức năng chính
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-button" id="surveillance-btn">
        <h3>Giám sát an ninh</h3>
        <p>Theo dõi và phát hiện đối tượng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cập", key="btn_surveillance"):
            st.session_state.menu_choice = "Giám sát an ninh"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-button" id="qr-btn">
        <h3>Quét mã QR</h3>
        <p>Quét mã QR để truy xuất thông tin công dân</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cập", key="btn_qr"):
            st.session_state.menu_choice = "Quét mã QR"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="feature-button" id="citizens-btn">
        <h3>Quản lý công dân</h3>
        <p>Xem và quản lý dữ liệu công dân</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cập", key="btn_citizens"):
            st.session_state.menu_choice = "Quản lý công dân"
            st.rerun()
    
    # Hiển thị thống kê
    st.markdown("### Thống kê hệ thống")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tổng số công dân", len(st.session_state.citizens_data))
    col_b.metric("Quét QR gần đây", "0")
    col_c.metric("Cảnh báo an ninh", "0")
    
    # Hiển thị hoạt động gần đây
    st.markdown("### Hoạt động gần đây")
    if len(st.session_state.citizens_data) > 0:
        recent_activities = st.session_state.citizens_data.sort_values(by='scan_date', ascending=False).head(5)
        for _, row in recent_activities.iterrows():
            st.text(f"{row['scan_date']} - Đã quét CCCD của {row['name']}")
    else:
        st.text("Chưa có hoạt động nào gần đây.")

# Hàm đăng xuất
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.menu_choice = "Trang chủ"
    st.rerun()

# Kiểm tra đăng nhập
if not st.session_state.logged_in:
    login_page()
else:
    # Hiển thị sidebar menu
    with st.sidebar:
        st.markdown(f"### Xin chào, {st.session_state.username}!")
        st.markdown("---")
        
        # Menu
        menu = ["Trang chủ", "Giám sát an ninh", "Quét mã QR", "Quản lý công dân"]
        choice = st.radio("Menu", menu, index=menu.index(st.session_state.menu_choice))
        
        if choice != st.session_state.menu_choice:
            st.session_state.menu_choice = choice
            st.rerun()
        
        st.markdown("---")
        if st.button("Đăng xuất"):
            logout()
    
    # Hiển thị trang tương ứng với lựa chọn menu
    if st.session_state.menu_choice == "Trang chủ":
        home_page()
    elif st.session_state.menu_choice == "Giám sát an ninh":
        surveillance_camera()
    elif st.session_state.menu_choice == "Quét mã QR":
        scan_qr_code()
    elif st.session_state.menu_choice == "Quản lý công dân":
        manage_citizens()