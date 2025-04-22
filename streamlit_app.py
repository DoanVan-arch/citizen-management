import streamlit as st
import cv2
import face_recognition
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os

# Khởi tạo session state
if 'citizens_data' not in st.session_state:
    st.session_state.citizens_data = pd.DataFrame(columns=['id', 'name', 'dob', 'address', 'scan_date'])

def scan_qr_code():
    st.subheader("Quét mã QR CCCD")
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Dừng quét")
    
    while not stop_button:
        ret, frame = cap.read()
        if ret:
            # Chuyển BGR sang RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Tìm và giải mã QR
            decoded_objects = decode(frame_rgb)
            
            for obj in decoded_objects:
                # Vẽ khung xung quanh mã QR
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(frame_rgb, [hull], True, (0, 255, 0), 2)
                
                # Giải mã dữ liệu
                qr_data = obj.data.decode('utf-8')
                
                # Xử lý dữ liệu từ QR (giả định format: id|name|dob|address)
                citizen_info = qr_data.split('|')
                if len(citizen_info) >= 4:
                    new_data = {
                        'id': citizen_info[0],
                        'name': citizen_info[1],
                        'dob': citizen_info[2],
                        'address': citizen_info[3],
                        'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.citizens_data = pd.concat([
                        st.session_state.citizens_data,
                        pd.DataFrame([new_data])
                    ], ignore_index=True)
                    st.success("Đã quét thành công!")
                    break
            
            frame_placeholder.image(frame_rgb, channels="RGB")
    
    cap.release()

def face_recognition_system():
    st.subheader("Nhận diện khuôn mặt")
    
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Dừng nhận diện")
    
    # Load known face encodings (trong thực tế, bạn sẽ load từ database)
    known_face_encodings = []
    known_face_names = []
    
    while not stop_button:
        ret, frame = cap.read()
        if ret:
            # Resize frame for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Tìm tất cả khuôn mặt trong frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Duyệt qua mỗi khuôn mặt tìm thấy trong frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Vẽ khung xung quanh khuôn mặt
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Nhận diện khuôn mặt (so sánh với known_face_encodings)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                # Hiển thị tên dưới khung
                cv2.putText(frame, name, (left, bottom + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
    
    cap.release()

def show_citizen_data():
    st.subheader("Dữ liệu công dân đã quét")
    if not st.session_state.citizens_data.empty:
        st.dataframe(st.session_state.citizens_data)
    else:
        st.info("Chưa có dữ liệu công dân nào.")

def main():
    st.title("Hệ thống Quản lý Công dân")
    
    menu = ["Trang chủ", "Quét QR CCCD", "Nhận diện khuôn mặt", "Xem dữ liệu"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Trang chủ":
        st.write("Chào mừng đến với hệ thống quản lý công dân")
        st.write("Vui lòng chọn chức năng từ menu bên trái")
        
    elif choice == "Quét QR CCCD":
        scan_qr_code()
        
    elif choice == "Nhận diện khuôn mặt":
        face_recognition_system()
        
    elif choice == "Xem dữ liệu":
        show_citizen_data()

if __name__ == '__main__':
    main()
