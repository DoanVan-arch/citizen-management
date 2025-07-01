from operator import truediv
import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os
import tempfile
from PIL import Image,ImageDraw
import av
from contextlib import contextmanager
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

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
        # Khởi tạo MTCNN cho Detect khuôn mặt
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # Ngưỡng Detect ba bước
            factor=0.709, 
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Khởi tạo FaceNet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.facenet = self.facenet.cuda()
            
        # Biến để lưu trữ embedding khuôn mặt
        self.known_face_embeddings = []
        self.known_face_names = []
        
        # Ngưỡng để xác định khuôn mặt giống nhau
        self.similarity_threshold = 0.6
        
        # Bộ đếm để tự động đặt tên cho khuôn mặt mới
        self.face_counter = 1

    def get_face_embedding(self, face_tensor):
        """Trích xuất embedding từ tensor khuôn mặt"""
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    face_tensor = face_tensor.cuda()
                embedding = self.facenet(face_tensor.unsqueeze(0))
                return F.normalize(embedding, p=2, dim=1)
        except:
            return None

    def find_matching_face(self, new_embedding):
        """Tìm khuôn mặt khớp trong danh sách Knowed"""
        if len(self.known_face_embeddings) == 0:
            return None, -1
        
        # Tính độ tương đồng với tất cả khuôn mặt Knowed
        similarities = []
        for known_embedding in self.known_face_embeddings:
            similarity = F.cosine_similarity(new_embedding, known_embedding).item()
            similarities.append(similarity)
        
        # Tìm độ tương đồng cao nhất
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        # Kiểm tra xem có vượt ngưỡng không
        if max_similarity > self.similarity_threshold:
            return self.known_face_names[max_index], max_similarity
        else:
            return None, max_similarity

    def add_new_face(self, embedding, name=None):
        """Thêm khuôn mặt mới vào danh sách"""
        if name is None:
            name = f"suspect {self.face_counter}"
            self.face_counter += 1
        
        self.known_face_embeddings.append(embedding)
        self.known_face_names.append(name)
        print(f"Đã thêm khuôn mặt mới: {name}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Chuyển đổi từ BGR sang RGB (MTCNN sử dụng RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect khuôn mặt bằng MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(rgb_img, landmarks=True)
        
        # Vẽ các khuôn mặt được Detect
        if boxes is not None:
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                # Lấy tọa độ khuôn mặt
                x1, y1, x2, y2 = [int(p) for p in box]
                
                # Vẽ hình chữ nhật xung quanh khuôn mặt
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ các điểm landmark (mắt, mũi, miệng)
                for p in landmark:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                
                # Trích xuất và xử lý khuôn mặt
                try:
                    # Cắt vùng khuôn mặt từ ảnh RGB
                    face_img = rgb_img[max(0, y1):min(rgb_img.shape[0], y2), 
                                     max(0, x1):min(rgb_img.shape[1], x2)]
                    
                    # Sử dụng MTCNN để chuẩn hóa khuôn mặt
                    face_tensor = self.mtcnn(face_img)
                    
                    if face_tensor is not None:
                        # Trích xuất embedding
                        embedding = self.get_face_embedding(face_tensor)
                        
                        if embedding is not None:
                            # Tìm khuôn mặt khớp
                            matched_name, similarity = self.find_matching_face(embedding)
                            
                            if matched_name:
                                # Khuôn mặt Knowed
                                label = f"{matched_name} ({similarity:.2f})"
                                color = (0, 255, 0)  # Xanh lá cho khuôn mặt Knowed
                            else:
                                # Khuôn mặt mới - thêm vào danh sách
                                self.add_new_face(embedding)
                                label = f"suspect {self.face_counter - 1} (New)"
                                color = (0, 0, 255)  # Đỏ cho khuôn mặt mới
                            
                            # Hiển thị tên/nhãn
                            cv2.putText(img, label, (x1, y1-30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                except Exception as e:
                    print(f"Lỗi xử lý khuôn mặt: {e}")
                
                # Hiển thị xác suất Detect
                confidence = f"Reliability: {probs[i]:.2f}"
                cv2.putText(img, confidence, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Hiển thị thống kê
        if boxes is not None:
            face_count = len(boxes)
            cv2.putText(img, f"detected: {face_count} | Knowed: {len(self.known_face_embeddings)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, f"detect: 0 | Knowed: {len(self.known_face_embeddings)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_known_faces_info(self):
        """Trả về thông tin các khuôn mặt Knowed"""
        return {
            "total_faces": len(self.known_face_embeddings),
            "face_names": self.known_face_names.copy()
        }
    
    def clear_known_faces(self):
        """Xóa tất cả khuôn mặt đã lưu"""
        self.known_face_embeddings.clear()
        self.known_face_names.clear()
        self.face_counter = 1
        print("Đã xóa tất cả khuôn mặt đã lưu")

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

@st.cache_resource
def load_face_models():
    # Khởi tạo MTCNN cho Detect khuôn mặt
    mtcnn = MTCNN(
        image_size=160, 
        margin=20, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709, 
        post_process=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Khởi tạo FaceNet model
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    if torch.cuda.is_available():
        facenet = facenet.cuda()
        
    return mtcnn, facenet

# Hàm Detect khuôn mặt trong ảnh
def detect_faces_in_image(image, mtcnn):
    # Chuyển đổi ảnh PIL sang numpy array nếu cần
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Chuyển đổi sang RGB nếu là BGR (từ OpenCV)
    if img_array.shape[2] == 3 and not isinstance(image, Image.Image):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Detect khuôn mặt
    boxes, probs, landmarks = mtcnn.detect(img_array, landmarks=True)
    
    return boxes, probs, landmarks, img_array

# Hàm vẽ kết quả Detect lên ảnh
def draw_faces_on_image(image, boxes, probs, landmarks):
    # Tạo bản sao để vẽ lên
    if isinstance(image, np.ndarray):
        # Nếu là numpy array, chuyển thành PIL Image
        result_image = Image.fromarray(image)
    else:
        # Nếu đã là PIL Image, tạo bản sao
        result_image = image.copy()
    
    draw = ImageDraw.Draw(result_image)
    
    # Vẽ các khuôn mặt được Detect
    if boxes is not None:
        for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
            # Vẽ hình chữ nhật xung quanh khuôn mặt
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                           outline="green", width=3)
            
            # Vẽ các điểm landmark
            for p in landmark:
                draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), 
                             fill="red")
            
            # Hiển thị độ tin cậy
            text_position = (box[0], box[1] - 15)
            confidence = f"Conf: {probs[i]:.2f}"
            draw.text(text_position, confidence, fill="green")
    
    # Hiển thị số lượng khuôn mặt
    if boxes is not None:
        face_count = len(boxes)
        draw.text((10, 10), f"Số khuôn mặt: {face_count}", fill="green")
    
    return result_image
ho_list = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Ngô", "Đinh"]
ten_dem_list = ["Văn", "Thị", "Minh", "Hữu", "Đức", "Thanh", "Quang", "Anh"]
ten_list = ["An", "Bình", "Cường", "Dung", "Em", "Phương", "Giang", "Hoa", "Inh", "Kim"]
import random
# Hàm xử lý video
def process_video(video_path, mtcnn, output_path=None):
    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Khởi tạo FaceNet model để trích xuất embedding
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    if torch.cuda.is_available():
        facenet = facenet.cuda()
    
    # Biến lưu trữ khuôn mặt Knowed
    known_face_embeddings = []
    known_face_names = []
    known_face_images = []
    face_counter = 1
    similarity_threshold = 0.6
    
    # Tạo video output nếu cần
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Tạo thanh tiến trình
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    face_count_per_frame = []
    
    def get_face_embedding(face_tensor):
        """Trích xuất embedding từ tensor khuôn mặt"""
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    face_tensor = face_tensor.cuda()
                embedding = facenet(face_tensor.unsqueeze(0))
                return F.normalize(embedding, p=2, dim=1)
        except:
            return None
    
    def find_matching_face(new_embedding):
        """Tìm khuôn mặt khớp trong danh sách Knowed"""
        if len(known_face_embeddings) == 0:
            return None, -1
        
        # Tính độ tương đồng với tất cả khuôn mặt Knowed
        similarities = []
        for known_embedding in known_face_embeddings:
            similarity = F.cosine_similarity(new_embedding, known_embedding).item()
            similarities.append(similarity)
        
        # Tìm độ tương đồng cao nhất
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        # Kiểm tra xem có vượt ngưỡng không
        if max_similarity > similarity_threshold:
            return known_face_names[max_index], max_similarity
        else:
            return None, max_similarity
   

    
    face_counter += 1
    def add_new_face(embedding,face_img, name=None):
        """Thêm khuôn mặt mới vào danh sách"""
        nonlocal face_counter
        if name is None:
            _random_citizen= get_random_citizen_info()
            if(_random_citizen==None):
                ho = random.choice(ho_list)
                ten_dem = random.choice(ten_dem_list)
                ten = random.choice(ten_list)
                name = f"{ho} {ten_dem} {ten}"
            else:
                
                name = _random_citizen['name']
    
    # Đảm bảo không trùng tên
            # while name in known_face_names:
            #     ho = random.choice(ho_list)
            #     ten_dem = random.choice(ten_dem_list)
            #     ten = random.choice(ten_list)
            #     name = f"{ho} {ten_dem} {ten}"
        if(name not in known_face_names):
            known_face_embeddings.append(embedding)
            known_face_names.append(name)
            known_face_images.append(face_img)
            print(f"Đã thêm khuôn mặt mới: {name}")
        return name
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển đổi từ BGR sang RGB cho MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect khuôn mặt
        boxes, probs, landmarks, _ = detect_faces_in_image(frame, mtcnn)
        
        # Lưu số lượng khuôn mặt
        if boxes is not None:
            face_count_per_frame.append(len(boxes))
        else:
            face_count_per_frame.append(0)
        
        # Vẽ kết quả lên frame
        if boxes is not None:
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                # Vẽ hình chữ nhật xung quanh khuôn mặt
                x1, y1, x2, y2 = [int(p) for p in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ các điểm landmark
                for p in landmark:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                
                # Xử lý nhận diện khuôn mặt
                try:
                    # Cắt vùng khuôn mặt từ ảnh RGB
                    face_img = rgb_frame[max(0, y1):min(rgb_frame.shape[0], y2), 
                                       max(0, x1):min(rgb_frame.shape[1], x2)]
                    
                    # Sử dụng MTCNN để chuẩn hóa khuôn mặt
                    face_tensor = mtcnn(face_img)
                    
                    if face_tensor is not None:
                        # Trích xuất embedding
                        embedding = get_face_embedding(face_tensor)
                        
                        if embedding is not None:
                            # Tìm khuôn mặt khớp
                            matched_name, similarity = find_matching_face(embedding)
                            
                            if matched_name:
                                # Khuôn mặt Knowed
                                label = f"{matched_name} ({similarity:.2f})"
                                name_color = (0, 255, 0)  # Xanh lá cho khuôn mặt Knowed
                            else:
                                # Khuôn mặt mới - thêm vào danh sách
                                new_name = add_new_face(embedding,face_img=face_img)
                                label = f"{new_name} (New)"
                                name_color = (0, 0, 255)  # Đỏ cho khuôn mặt mới
                            
                            # Hiển thị tên/nhãn
                            cv2.putText(frame, label, (x1, y1-30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 2)
                
                except Exception as e:
                    print(f"Lỗi xử lý khuôn mặt: {e}")
                    # Hiển thị nhãn mặc định nếu có lỗi
                    cv2.putText(frame, "Unknown", (x1, y1-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Hiển thị độ tin cậy
                confidence = f"Conf: {probs[i]:.2f}"
                cv2.putText(frame, confidence, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị thống kê
        if boxes is not None:
            face_count = len(boxes)
            stats_text = f"Detect: {face_count} | Knowed: {len(known_face_embeddings)}"
            cv2.putText(frame, stats_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            stats_text = f"Detect: 0 | Knowed: {len(known_face_embeddings)}"
            cv2.putText(frame, stats_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Ghi frame vào video output nếu cần
        if output_path:
            out.write(frame)
        
        # Cập nhật tiến trình
        frame_count += 1
        progress = int(frame_count / total_frames * 100)
        progress_bar.progress(progress / 100)
        status_text.text(f"Đang xử lý: {progress}% ({frame_count}/{total_frames}) - Đã nhận diện: {len(known_face_embeddings)} người")
    
    # Giải phóng tài nguyên
    cap.release()
    if output_path:
        out.release()
    
    # Tính toán thống kê
    max_faces = max(face_count_per_frame) if face_count_per_frame else 0
    avg_faces = sum(face_count_per_frame) / len(face_count_per_frame) if face_count_per_frame else 0
    
    # Hiển thị danh sách người đã nhận diện
    st.success(f"Hoàn thành xử lý video!")
    st.info(f"Đã nhận diện {len(known_face_embeddings)} người khác nhau:")
    for i, name in enumerate(known_face_names, 1):
        st.write(f"{i}. {name}")
    
    return {
        "total_frames": frame_count,
        "max_faces": max_faces,
        "avg_faces": avg_faces,
        "face_count_per_frame": face_count_per_frame,
        "known_faces": len(known_face_embeddings),
        "known_face_names": known_face_names.copy(),
        "known_face_embeddings": known_face_embeddings.copy(),
        "known_face_images":known_face_images.copy()
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
        <p>Theo dõi và Detect đối tượng qua camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to choose between aiortc and file upload
        camera_option = st.radio(
            "Chọn phương thức:",
            ["Camera trực tiếp", "Upload video/ảnh"],
            key="camera_option"
        )
        
        if camera_option == "Camera trực tiếp":
            try:
                if AIORTC_AVAILABLE:
                    response = requests.get(
                        "https://iewcom1.metered.live/api/v1/turn/credentials",
                        params={"apiKey": "097b76f5eee1b5486c8b495410bd84adf5f2"}
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
            mtcnn, facenet = load_face_models()
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
                        with st.spinner("Đang phân tích ảnh..."):
                            # Detect khuôn mặt
                            boxes, probs, landmarks, img_array = detect_faces_in_image(image, mtcnn)
                            
                            # Vẽ kết quả
                            result_image = draw_faces_on_image(image, boxes, probs, landmarks)
                            
                            # Hiển thị kết quả
                            st.image(result_image, caption="Kết quả phân tích", use_container_width=True)
                            
                            # Hiển thị thông tin
                            if boxes is not None:
                                st.success(f"Detected {len(boxes)} face")
                                
                                # Hiển thị thông tin chi tiết cho mỗi khuôn mặt
                                for i, (box, prob) in enumerate(zip(boxes, probs)):
                                    with st.expander(f"Face #{i+1} (Relibity: {prob:.2f})"):
                                        # Cắt khuôn mặt từ ảnh gốc
                                        x1, y1, x2, y2 = [int(p) for p in box]
                                        face_img = Image.fromarray(img_array[y1:y2, x1:x2])
                                        
                                        # Hiển thị khuôn mặt đã cắt
                                        st.image(face_img, caption=f"face #{i+1}", width=150)
                                        
                                        # Hiển thị thông tin vị trí
                                        st.text(f"Vị trí: X1={x1}, Y1={y1}, X2={x2}, Y2={y2}")
                            else:
                                st.warning("Không Detect khuôn mặt nào trong ảnh")
                
                else:  # Video file
                    st.video(uploaded_file)
                    
                    if st.button("Phân tích video"):
                        # Lưu video tạm thời để xử lý
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, "input_video.mp4")
                        output_path = os.path.join(temp_dir, "output_video.mp4")
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        with st.spinner("Đang phân tích video... Quá trình này có thể mất vài phút"):
                            # Xử lý video
                            results = process_video(temp_path, mtcnn, output_path)
                            
                            # Hiển thị video đã xử lý
                            st.success("Phân tích hoàn tất!")
                            
                            # Hiển thị thống kê tổng quan
                            st.subheader("📊 Thống kê tổng quan")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Tổng số frame", results["total_frames"])
                            with col2:
                                st.metric("Số khuôn mặt tối đa", int(results["max_faces"]))
                            with col3:
                                st.metric("Số khuôn mặt trung bình", f"{results['avg_faces']:.2f}")
                            with col4:
                                st.metric("Số người đã nhận diện", results.get("known_faces", 0))
                            
                            # Hiển thị thông tin chi tiết về khuôn mặt đã tìm được
                            if results.get("known_faces", 0) > 0:
                                st.subheader("👥 Danh sách người đã nhận diện")
                                
                                # Tạo tabs cho từng người
                                if len(results.get("known_face_names", [])) > 0:
                                    tabs = st.tabs([f"👤 {name}" for name in results["known_face_names"]])
                                    
                                    for i, (tab, name) in enumerate(zip(tabs, results["known_face_names"])):
                                        with tab:
                                            col_info, col_image = st.columns([2, 1])
                                            
                                            with col_info:
                                                st.write(f"**Tên:** {name}")
                                                st.write(f"**ID:** {random.randint(100000000000,999999999999)}")
                                                st.write(f"**Trạng thái:** Đã nhận diện")
                                                
                                                # Hiển thị thông tin embedding (tùy chọn)
                                                if "known_face_embeddings" in results and i < len(results["known_face_embeddings"]):
                                                    embedding = results["known_face_embeddings"][i]
                                                    st.write(f"**Kích thước embedding:** {embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
                                            
                                            with col_image:
                                                # Placeholder cho ảnh khuôn mặt (sẽ cần thêm logic để lưu ảnh khuôn mặt)
                                                st.info("Ảnh khuôn mặt sẽ được hiển thị ở đây")
                                                st.image(results["known_face_images"][i], caption=f"Khuôn mặt của {name}", width=150)
                                
                                # Hiển thị bảng tóm tắt
                                st.subheader("📋 Bảng tóm tắt")
                                face_data = []
                                for i, name in enumerate(results.get("known_face_names", [])):
                                    face_data.append({
                                        "STT": i + 1,
                                        "Tên": name,
                                        "Trạng thái": "Đã nhận diện",
                                        "Lần xuất hiện": "Nhiều lần"  # Có thể tính toán chính xác hơn
                                    })
                                
                                if face_data:
                                    import pandas as pd
                                    df = pd.DataFrame(face_data)
                                    st.dataframe(df, use_container_width=True)
                            
                            else:
                                st.info("Không tìm thấy khuôn mặt nào trong video.")
                            
                            # Hiển thị biểu đồ số lượng khuôn mặt theo frame
                            if results.get("face_count_per_frame"):
                                st.subheader("📈 Biểu đồ số lượng khuôn mặt theo thời gian")
                                
                                
                                
                                fig, ax = plt.subplots(figsize=(12, 4))
                                frames = range(len(results["face_count_per_frame"]))
                                ax.plot(frames, results["face_count_per_frame"], linewidth=1, alpha=0.7)
                                ax.fill_between(frames, results["face_count_per_frame"], alpha=0.3)
                                ax.set_xlabel("Frame")
                                ax.set_ylabel("Số lượng khuôn mặt")
                                ax.set_title("Số lượng khuôn mặt phát hiện theo từng frame")
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                            
                            # Hiển thị video đã xử lý
                            st.subheader("🎥 Video đã xử lý")
                            if os.path.exists(output_path):
                                # Nút tải xuống
                                with open(output_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Tải xuống video đã xử lý",
                                        data=file,
                                        file_name="face_detection_result.mp4",
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                                
                                # Hiển thị video
                                st.video(output_path)
                            
                            # Thêm thông tin chi tiết (có thể thu gọn)
                            with st.expander("🔍 Thông tin chi tiết"):
                                st.json({
                                    "Tổng số frame": results["total_frames"],
                                    "Số khuôn mặt tối đa trong 1 frame": int(results["max_faces"]),
                                    "Số khuôn mặt trung bình": round(results["avg_faces"], 2),
                                    "Số người đã nhận diện": results.get("known_faces", 0),
                                    "Danh sách tên": results.get("known_face_names", [])
                                })
                            
                            # Dọn dẹp file tạm
                            try:
                                import shutil
                                shutil.rmtree(temp_dir)
                            except:
                                pass

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Điều khiển Camera</h3>
        <p>Cài đặt và điều khiển camera giám sát</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        detection_options = st.multiselect(
            "Chọn các đối tượng cần Detect:",
            ["Khuôn mặt"],
            default=["Khuôn mặt"],
            key="detection_options"
        )
        
        sensitivity = st.slider("Độ nhạy Detect", 0, 100, 50, key="sensitivity")
        
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
JSON_FILE_PATH ="data.json"
def load_data_from_json():
    """
    Đọc dữ liệu từ file JSON
    """
    try:
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Chuyển đổi từ JSON sang DataFrame
            if data:  # Nếu có dữ liệu
                df = pd.DataFrame(data)
                return df
            else:
                # Tạo DataFrame rỗng với các cột cần thiết
                return pd.DataFrame(columns=[
                    'id', 'cccd', 'name', 'dob', 'sex', 
                    'address', 'expdate', 'scan_date', 'image_path'
                ])
        else:
            # Tạo DataFrame rỗng nếu file không tồn tại
            return pd.DataFrame(columns=[
                'id', 'cccd', 'name', 'dob', 'sex', 
                'address', 'expdate', 'scan_date', 'image_path'
            ])
    
    except Exception as e:
        st.error(f"Lỗi khi đọc file JSON: {str(e)}")
        return pd.DataFrame(columns=[
            'id', 'cccd', 'name', 'dob', 'sex', 
            'address', 'expdate', 'scan_date', 'image_path'
        ])

def save_data_to_json(dataframe):
    """
    Lưu DataFrame vào file JSON
    """
    try:
        # Chuyển đổi DataFrame sang dictionary
        data_dict = dataframe.to_dict('records')
        
        # Lưu vào file JSON
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        return True, "Dữ liệu đã được lưu thành công!"
    
    except Exception as e:
        return False, f"Lỗi khi lưu file JSON: {str(e)}"

def initialize_session_state():
    """
    Khởi tạo session state với dữ liệu từ JSON
    """
    
    st.session_state.citizens_data = load_data_from_json()

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
                
                # Lưu dữ liệu vào JSON ngay sau khi thêm
                success, message = save_data_to_json(st.session_state.citizens_data)
                if not success:
                    st.warning(f"Cảnh báo: {message}")
                
                return True, "QR code processed successfully!"
                
        return False, "Không tìm thấy QR code hợp lệ."
    
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

def delete_citizen_record(index):
    """
    Xóa một bản ghi công dân
    """
    try:
        # Xóa ảnh nếu tồn tại
        if index < len(st.session_state.citizens_data):
            image_path = st.session_state.citizens_data.iloc[index]['image_path']
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Xóa bản ghi khỏi DataFrame
        st.session_state.citizens_data = st.session_state.citizens_data.drop(index).reset_index(drop=True)
        
        # Lưu lại vào JSON
        success, message = save_data_to_json(st.session_state.citizens_data)
        return success, message if success else f"Đã xóa bản ghi nhưng có lỗi khi lưu: {message}"
        
    except Exception as e:
        return False, f"Lỗi khi xóa bản ghi: {str(e)}"



def get_random_citizen_info():
    """Lấy thông tin công dân ngẫu nhiên"""
    try:
        # Kiểm tra session state có dữ liệu không
        if 'citizens_data' not in st.session_state:
            print("Không có dữ liệu công dân trong session state")
            return None
        
        if st.session_state.citizens_data is None or st.session_state.citizens_data.empty:
            print("Dữ liệu công dân rỗng")
            return None
        
        # Lấy ngẫu nhiên một citizen
        random_citizen = st.session_state.citizens_data.sample(n=1).iloc[0]
        
        print(f"Đã chọn ngẫu nhiên công dân: {random_citizen.get('name', 'Unknown')}")
        return random_citizen
        
    except Exception as e:
        print(f"Lỗi khi lấy citizen ngẫu nhiên: {e}")
        return None

def clear_all_data():
    """
    Xóa tất cả dữ liệu
    """
    try:
        # Xóa tất cả ảnh
        if len(st.session_state.citizens_data) > 0:
            for _, row in st.session_state.citizens_data.iterrows():
                if os.path.exists(row['image_path']):
                    os.remove(row['image_path'])
        
        # Tạo DataFrame rỗng
        st.session_state.citizens_data = pd.DataFrame(columns=[
            'id', 'cccd', 'name', 'dob', 'sex', 
            'address', 'expdate', 'scan_date', 'image_path'
        ])
        
        # Lưu DataFrame rỗng vào JSON
        success, message = save_data_to_json(st.session_state.citizens_data)
        return success, "Đã xóa tất cả dữ liệu!"
        
    except Exception as e:
        return False, f"Lỗi khi xóa dữ liệu: {str(e)}"
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
    tab1, tab2,tab3 = st.tabs(["📁 Upload Ảnh", "📷 Camera WebRTC"," Giới thiệu công nghệ"])
    
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
                    "https://iewcom1.metered.live/api/v1/turn/credentials",
                    params={"apiKey": "097b76f5eee1b5486c8b495410bd84adf5f2"}
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
                    <li>Hệ thống sẽ tự động Detect và xử lý QR code</li>
                    <li>QR code được Detect sẽ có khung màu xanh</li>
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
                            st.success("🎉 QR Code đã được Detect!")
                            
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
    with tab3:
        st.markdown("""
        ### MTCNN (Multi-task Cascaded Convolutional Networks)
        
        MTCNN là một thuật toán Detect khuôn mặt hiệu quả cao, hoạt động thông qua 3 giai đoạn cascade:
        
        1. **P-Net (Proposal Network)**: Tạo các hộp đề xuất ban đầu
        2. **R-Net (Refinement Network)**: Tinh chỉnh các hộp đề xuất
        3. **O-Net (Output Network)**: Tạo kết quả cuối cùng với các điểm đặc trưng (landmarks)
        
        MTCNN không chỉ Detect khuôn mặt mà còn xác định 5 điểm đặc trưng quan trọng: 2 mắt, mũi và 2 góc miệng.
        
        ### FaceNet
        
        FaceNet là một mô hình học sâu được Google phát triển, chuyển đổi khuôn mặt thành vector đặc trưng 128 chiều.
        Mô hình này có thể được sử dụng để:
        
        - Nhận dạng khuôn mặt
        - Xác minh khuôn mặt (kiểm tra xem hai khuôn mặt có phải là cùng một người)
        - Phân cụm khuôn mặt
        
        Trong ứng dụng này, chúng tôi sử dụng MTCNN để Detect khuôn mặt và có thể mở rộng với FaceNet để nhận dạng.
        """,unsafe_allow_html=True)
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
            st.success("✅ QR code đã được Detect và xử lý thành công!")
            
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
        st.warning("Detect nhiều lỗi aiortc. Đang reset session...")
        for key in list(st.session_state.keys()):
            if 'aiortc' in key.lower() or 'peer' in key.lower():
                del st.session_state[key]
        st.session_state.aiortc_error_count = 0
        st.rerun()

def show_citizen_data():
    st.markdown("<h2 style='text-align: center;'>Dữ liệu Công dân</h2>", unsafe_allow_html=True)
    
    if not st.session_state.citizens_data.empty:
        # Header với thống kê và nút xóa hết
        col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
        
        with col_header1:
            st.info(f"📊 Tổng số công dân: **{len(st.session_state.citizens_data)}**")
        
        with col_header2:
            # Nút backup trước khi xóa
            if st.button("💾 Backup dữ liệu", type="secondary"):
                success, message = export_data_to_json()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col_header3:
            # Nút xóa tất cả với confirmation
            if st.button("🗑️ Xóa tất cả", type="secondary"):
                st.session_state.show_delete_all_confirm = True
        
        # Confirmation dialog cho xóa tất cả
        if getattr(st.session_state, 'show_delete_all_confirm', False):
            st.warning("⚠️ **Cảnh báo:** Bạn có chắc chắn muốn xóa tất cả dữ liệu?")
            col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 2])
            
            with col_confirm1:
                if st.button("✅ Xác nhận xóa", type="primary"):
                    success, message = clear_all_data()
                    if success:
                        st.success(message)
                        st.session_state.show_delete_all_confirm = False
                        st.rerun()
                    else:
                        st.error(message)
            
            with col_confirm2:
                if st.button("❌ Hủy bỏ"):
                    st.session_state.show_delete_all_confirm = False
                    st.rerun()
        
        st.divider()
        
        # Hiển thị từng bản ghi với nút xóa
        for index, row in st.session_state.citizens_data.iterrows():
            with st.expander(f"Công dân: {row['name']} - CCCD: {row['cccd']}"):
                # Layout chính
                col1, col2, col3 = st.columns([1, 2, 0.5])
                
                with col1:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="Ảnh CCCD", use_container_width=True)
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
                
                with col3:
                    st.markdown("**Thao tác:**")
                    
                    # Nút xóa từng bản ghi
                    if st.button(f"🗑️ Xóa", key=f"delete_single_{index}", type="secondary"):
                        st.session_state[f'show_delete_confirm_{index}'] = True
                    
                    # Confirmation cho xóa từng bản ghi
                    if getattr(st.session_state, f'show_delete_confirm_{index}', False):
                        st.warning("⚠️ Xác nhận xóa?")
                        
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("✅", key=f"confirm_delete_{index}", type="primary"):
                                success, message = delete_citizen_record(index)
                                if success:
                                    st.success(message)
                                    # Reset confirmation state
                                    st.session_state[f'show_delete_confirm_{index}'] = False
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        with col_del2:
                            if st.button("❌", key=f"cancel_delete_{index}"):
                                st.session_state[f'show_delete_confirm_{index}'] = False
                                st.rerun()
                
                # Thêm thông tin bổ sung (tùy chọn)
                with st.container():
                    st.markdown("---")
                    col_extra1, col_extra2, col_extra3 = st.columns(3)
                    
                    with col_extra1:
                        st.caption(f"🕒 Thời gian quét: {row['scan_date']}")
                    
                    with col_extra2:
                        # Tính tuổi từ ngày sinh
                        try:
                            dob = datetime.strptime(row['dob'], "%d/%m/%Y")
                            age = datetime.now().year - dob.year
                            st.caption(f"🎂 Tuổi: {age}")
                        except:
                            st.caption("🎂 Tuổi: N/A")
                    
                    with col_extra3:
                        # Kiểm tra hạn CCCD
                        try:
                            exp_date = datetime.strptime(row['expdate'], "%d/%m/%Y")
                            days_left = (exp_date - datetime.now()).days
                            if days_left < 0:
                                st.caption("⚠️ **Đã hết hạn**")
                            elif days_left < 30:
                                st.caption(f"⚠️ Còn {days_left} ngày")
                            else:
                                st.caption(f"✅ Còn {days_left} ngày")
                        except:
                            st.caption("📅 Hạn: N/A")
                
    else:
        # Thông báo khi không có dữ liệu
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h3>📋 Chưa có dữ liệu công dân nào</h3>
            <p>Hãy quét QR code để thêm thông tin công dân mới</p>
        </div>
        """, unsafe_allow_html=True)

def export_data_to_json(filename=None):
    """
    Xuất dữ liệu ra file JSON khác (backup)
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"citizens_backup_{timestamp}.json"
        
        data_dict = st.session_state.citizens_data.to_dict('records')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        return True, f"Dữ liệu đã được backup vào file: {filename}"
    
    except Exception as e:
        return False, f"Lỗi khi backup dữ liệu: {str(e)}"

# Thêm CSS để làm đẹp giao diện
def add_custom_css():
    st.markdown("""
    <style>
    /* Style cho nút xóa */
    .stButton > button[kind="secondary"] {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #ff6b6b;
        color: white;
    }
    
    /* Style cho confirmation buttons */
    .stButton > button[kind="primary"] {
        background-color: #00cc88;
        color: white;
        border: none;
        border-radius: 5px;
    }
    
    /* Style cho expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* Style cho warning messages */
    .stAlert > div {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    
    
    # Khởi tạo session state
    initialize_session_state()
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