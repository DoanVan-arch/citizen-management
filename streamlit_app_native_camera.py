from operator import truediv
import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import tempfile

# Thiu1ebft lu1eadp giao diu1ec7n trang
st.set_page_config(
    page_title="Hu1ec6 THu1ed0NG QUu1ea2N Lu00dd Cu00d4NG Du00c2N",
    page_icon="ud83dudccb",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Danh su00e1ch tu00e0i khou1ea3n mu1eabu (trong thu1ef1c tu1ebf nu00ean lu01b0u trong cu01a1 su1edf du1eef liu1ec7u vu00e0 mu00e3 hu00f3a mu1eadt khu1ea9u)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

# Hu00e0m xu1eed lu00fd u1ea3nh u0111u1ec3 phu00e1t hiu1ec7n QR code
def process_image_for_qr(image):
    """Xu1eed lu00fd u1ea3nh u0111u1ec3 tu00ecm vu00e0 giu1ea3i mu00e3 QR code"""
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
                
        return False, "Khu00f4ng tu00ecm thu1ea5y mu00e3 QR trong u1ea3nh."
    
    except Exception as e:
        return False, f"Lu1ed7i: {str(e)}"

# Hu00e0m u0111u0103ng nhu1eadp
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

# Hu00e0m giu00e1m su00e1t camera
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
        
        # Add option to choose between camera and file upload
        camera_option = st.radio(
            "Chu1ecdn phu01b0u01a1ng thu1ee9c:",
            ["Camera tru1ef1c tiu1ebfp", "Upload video/u1ea3nh"],
            key="camera_option"
        )
        
        if camera_option == "Camera tru1ef1c tiu1ebfp":
            try:
                # Use Streamlit's native camera input
                camera_image = st.camera_input("Camera giu00e1m su00e1t", key="surveillance_camera")
                
                if camera_image is not None:
                    # Process the captured image
                    image = Image.open(camera_image)
                    image_array = np.array(image)
                    
                    # Flip the image horizontally for a mirror effect
                    flipped_image = cv2.flip(image_array, 1)
                    
                    # Display the processed image
                    st.image(flipped_image, caption="Camera Feed", use_container_width=True)
                    
                    # Save button
                    if st.button("Lu01b0u u1ea3nh", key="save_surveillance"):
                        # Create temp directory if it doesn't exist
                        if not os.path.exists("captured_images"):
                            os.makedirs("captured_images")
                        
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_images/surveillance_{timestamp}.jpg"
                        Image.fromarray(flipped_image).save(filename)
                        st.success(f"u1ea2nh u0111u00e3 u0111u01b0u1ee3c lu01b0u tu1ea1i {filename}")
                        
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
                    # Save video to temp file and process
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    
                    # Display video
                    st.video(tfile.name)
                    
                    if st.button("Phu00e2n tu00edch video"):
                        st.success("u0110ang phu00e2n tu00edch video...")
                        # Add your video analysis logic here
                    
                    # Clean up temp file
                    os.unlink(tfile.name)
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Thu00f4ng tin giu00e1m su00e1t</h3>
        <p>Du1eef liu1ec7u vu00e0 cu1ea3nh bu00e1o tu1eeb hu1ec7 thu1ed1ng giu00e1m su00e1t</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display some mock surveillance data
        st.markdown("### Tru1ea1ng thu00e1i hu1ec7 thu1ed1ng")
        st.success("u2705 Hu1ec7 thu1ed1ng u0111ang hou1ea1t u0111u1ed9ng bu00ecnh thu01b0u1eddng")
        
        st.markdown("### Thu1ed1ng ku00ea")
        col_a, col_b = st.columns(2)
        col_a.metric("u0110u1ed1i tu01b0u1ee3ng phu00e1t hiu1ec7n", "0")
        col_b.metric("Cu1ea3nh bu00e1o", "0")
        
        st.markdown("### Nhu1eadt ku00fd hou1ea1t u0111u1ed9ng")
        st.text("10:30:45 - Khu1edfi u0111u1ed9ng hu1ec7 thu1ed1ng")
        st.text("10:31:12 - Ku1ebft nu1ed1i camera thu00e0nh cu00f4ng")
        st.text(f"{datetime.now().strftime('%H:%M:%S')} - u0110ang giu00e1m su00e1t...")

# Hu00e0m quu00e9t mu00e3 QR
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
        
        # Use Streamlit's native camera input
        camera_image = st.camera_input("Quu00e9t mu00e3 QR", key="qr_camera")
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            image_array = np.array(image)
            
            # Display the image
            st.image(image_array, caption="u1ea2nh u0111u00e3 chu1ee5p", use_container_width=True)
            
            # Process for QR code
            if st.button("Xu1eed lu00fd QR Code", key="process_camera_qr"):
                with st.spinner("u0110ang xu1eed lu00fd..."):
                    success, message = process_image_for_qr(image)
                    if success:
                        st.success(message)
                    else:
                        st.error("Khu00f4ng tu00ecm thu1ea5y mu00e3 QR trong u1ea3nh. Vui lu00f2ng thu1eed lu1ea1i.")

# Hu00e0m quu1ea3n lu00fd du1eef liu1ec7u cu00f4ng du00e2n
def manage_citizens():
    st.markdown("<h1 style='text-align: center;'>Quu1ea3n lu00fd du1eef liu1ec7u cu00f4ng du00e2n</h1>", unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb du1eef liu1ec7u cu00f4ng du00e2n
    if len(st.session_state.citizens_data) > 0:
        st.markdown("""
        <div class="info-card">
        <h3>Danh su00e1ch cu00f4ng du00e2n</h3>
        <p>Thu00f4ng tin cu00f4ng du00e2n u0111u00e3 u0111u01b0u1ee3c lu01b0u trong hu1ec7 thu1ed1ng</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiu1ec3n thu1ecb bu1ea3ng du1eef liu1ec7u
        st.dataframe(st.session_state.citizens_data[['id', 'cccd', 'name', 'dob', 'sex', 'address', 'scan_date']])
        
        # Chu1ecdn cu00f4ng du00e2n u0111u1ec3 xem chi tiu1ebft
        citizen_ids = st.session_state.citizens_data['id'].tolist()
        selected_id = st.selectbox("Chu1ecdn ID cu00f4ng du00e2n u0111u1ec3 xem chi tiu1ebft", citizen_ids)
        
        if selected_id:
            # Lu1ea5y thu00f4ng tin cu00f4ng du00e2n
            citizen = st.session_state.citizens_data[st.session_state.citizens_data['id'] == selected_id].iloc[0]
            
            st.markdown("### Thu00f4ng tin chi tiu1ebft cu00f4ng du00e2n")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if citizen['image_path'] and os.path.exists(citizen['image_path']):
                    st.image(citizen['image_path'], caption="u1ea3nh CCCD", use_container_width=True)
                else:
                    st.info("Khu00f4ng cu00f3 u1ea3nh")
            
            with col2:
                st.markdown(f"**Hu1ecd tu00ean:** {citizen['name']}")
                st.markdown(f"**CCCD:** {citizen['cccd']}")
                st.markdown(f"**Ngu00e0y sinh:** {citizen['dob']}")
                st.markdown(f"**Giu1edbi tu00ednh:** {citizen['sex']}")
                st.markdown(f"**u0110u1ecba chu1ec9:** {citizen['address']}")
                st.markdown(f"**Ngu00e0y hu1ebft hu1ea1n:** {citizen['expdate']}")
                st.markdown(f"**Ngu00e0y quu00e9t:** {citizen['scan_date']}")
                
                # Nu00fat xu00f3a cu00f4ng du00e2n
                if st.button("Xu00f3a cu00f4ng du00e2n nu00e0y"):
                    st.session_state.citizens_data = st.session_state.citizens_data[st.session_state.citizens_data['id'] != selected_id]
                    st.success("u0110u00e3 xu00f3a thu00f4ng tin cu00f4ng du00e2n!")
                    st.rerun()
    else:
        st.info("Chu01b0a cu00f3 du1eef liu1ec7u cu00f4ng du00e2n nu00e0o. Vui lu00f2ng quu00e9t mu00e3 QR u0111u1ec3 thu00eam cu00f4ng du00e2n.")
    
    # Thu00eam cu00f4ng du00e2n mu1edbi thu1ee7 cu00f4ng
    st.markdown("""
    <div class="info-card">
    <h3>Thu00eam cu00f4ng du00e2n mu1edbi</h3>
    <p>Nhu1eadp thu00f4ng tin cu00f4ng du00e2n mu1edbi vu00e0o hu1ec7 thu1ed1ng</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_cccd = st.text_input("Su1ed1 CCCD")
        new_name = st.text_input("Hu1ecd tu00ean")
        new_dob = st.date_input("Ngu00e0y sinh")
        new_sex = st.selectbox("Giu1edbi tu00ednh", ["Nam", "Nu1eef"])
    
    with col2:
        new_address = st.text_area("u0110u1ecba chu1ec9")
        new_expdate = st.date_input("Ngu00e0y hu1ebft hu1ea1n")
        new_image = st.file_uploader("Tu1ea3i lu00ean u1ea3nh CCCD", type=["jpg", "jpeg", "png"])
    
    if st.button("Thu00eam cu00f4ng du00e2n"):
        if new_cccd and new_name:
            # Tu1ea1o ID mu1edbi
            new_id = len(st.session_state.citizens_data) + 1
            
            # Xu1eed lu00fd u1ea3nh nu1ebfu cu00f3
            image_path = None
            if new_image:
                if not os.path.exists("citizen_images"):
                    os.makedirs("citizen_images")
                
                image_path = f"citizen_images/citizen_{new_cccd}.jpg"
                Image.open(new_image).save(image_path)
            
            # Tu1ea1o du1eef liu1ec7u cu00f4ng du00e2n mu1edbi
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
            
            # Thu00eam vu00e0o dataframe
            st.session_state.citizens_data = pd.concat([
                st.session_state.citizens_data, 
                pd.DataFrame([new_citizen])
            ], ignore_index=True)
            
            st.success(f"u0110u00e3 thu00eam cu00f4ng du00e2n {new_name} vu00e0o hu1ec7 thu1ed1ng!")
            st.rerun()
        else:
            st.error("Vui lu00f2ng nhu1eadp u0111u1ea7y u0111u1ee7 thu00f4ng tin bu1eaft buu1ed9c (CCCD vu00e0 Hu1ecd tu00ean)")

# Hu00e0m trang chu1ee7
def home_page():
    st.markdown("<h1 style='text-align: center;'>Hu1ec6 THu1ed0NG QUu1ea2N Lu00dd Cu00d4NG Du00c2N</h1>", unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb thu00f4ng tin ngu01b0u1eddi du00f9ng
    st.markdown(f"""
    <div class="info-card">
    <h3>Xin chu00e0o, {st.session_state.username}!</h3>
    <p>Chu00e0o mu1eebng bu1ea1n u0111u1ebfn vu1edbi Hu1ec7 thu1ed1ng Quu1ea3n lu00fd Cu00f4ng du00e2n. Vui lu00f2ng chu1ecdn chu1ee9c nu0103ng bu00ean du01b0u1edbi u0111u1ec3 bu1eaft u0111u1ea7u.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hiu1ec3n thu1ecb cu00e1c chu1ee9c nu0103ng chu00ednh
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-button" id="surveillance-btn">
        <h3>Giu00e1m su00e1t an ninh</h3>
        <p>Theo du00f5i vu00e0 phu00e1t hiu1ec7n u0111u1ed1i tu01b0u1ee3ng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cu1eadp", key="btn_surveillance"):
            st.session_state.menu_choice = "Giu00e1m su00e1t an ninh"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-button" id="qr-btn">
        <h3>Quu00e9t mu00e3 QR</h3>
        <p>Quu00e9t mu00e3 QR u0111u1ec3 truy xuu1ea5t thu00f4ng tin cu00f4ng du00e2n</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cu1eadp", key="btn_qr"):
            st.session_state.menu_choice = "Quu00e9t mu00e3 QR"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="feature-button" id="citizens-btn">
        <h3>Quu1ea3n lu00fd cu00f4ng du00e2n</h3>
        <p>Xem vu00e0 quu1ea3n lu00fd du1eef liu1ec7u cu00f4ng du00e2n</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Truy cu1eadp", key="btn_citizens"):
            st.session_state.menu_choice = "Quu1ea3n lu00fd cu00f4ng du00e2n"
            st.rerun()
    
    # Hiu1ec3n thu1ecb thu1ed1ng ku00ea
    st.markdown("### Thu1ed1ng ku00ea hu1ec7 thu1ed1ng")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tu1ed5ng su1ed1 cu00f4ng du00e2n", len(st.session_state.citizens_data))
    col_b.metric("Quu00e9t QR gu1ea7n u0111u00e2y", "0")
    col_c.metric("Cu1ea3nh bu00e1o an ninh", "0")
    
    # Hiu1ec3n thu1ecb hou1ea1t u0111u1ed9ng gu1ea7n u0111u00e2y
    st.markdown("### Hou1ea1t u0111u1ed9ng gu1ea7n u0111u00e2y")
    if len(st.session_state.citizens_data) > 0:
        recent_activities = st.session_state.citizens_data.sort_values(by='scan_date', ascending=False).head(5)
        for _, row in recent_activities.iterrows():
            st.text(f"{row['scan_date']} - u0110u00e3 quu00e9t CCCD cu1ee7a {row['name']}")
    else:
        st.text("Chu01b0a cu00f3 hou1ea1t u0111u1ed9ng nu00e0o gu1ea7n u0111u00e2y.")

# Hu00e0m u0111u0103ng xuu1ea5t
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.menu_choice = "Trang chu1ee7"
    st.rerun()

# Kiu1ec3m tra u0111u0103ng nhu1eadp
if not st.session_state.logged_in:
    login_page()
else:
    # Hiu1ec3n thu1ecb sidebar menu
    with st.sidebar:
        st.markdown(f"### Xin chu00e0o, {st.session_state.username}!")
        st.markdown("---")
        
        # Menu
        menu = ["Trang chu1ee7", "Giu00e1m su00e1t an ninh", "Quu00e9t mu00e3 QR", "Quu1ea3n lu00fd cu00f4ng du00e2n"]
        choice = st.radio("Menu", menu, index=menu.index(st.session_state.menu_choice))
        
        if choice != st.session_state.menu_choice:
            st.session_state.menu_choice = choice
            st.rerun()
        
        st.markdown("---")
        if st.button("u0110u0103ng xuu1ea5t"):
            logout()
    
    # Hiu1ec3n thu1ecb trang tu01b0u01a1ng u1ee9ng vu1edbi lu1ef1a chu1ecdn menu
    if st.session_state.menu_choice == "Trang chu1ee7":
        home_page()
    elif st.session_state.menu_choice == "Giu00e1m su00e1t an ninh":
        surveillance_camera()
    elif st.session_state.menu_choice == "Quu00e9t mu00e3 QR":
        scan_qr_code()
    elif st.session_state.menu_choice == "Quu1ea3n lu00fd cu00f4ng du00e2n":
        manage_citizens()