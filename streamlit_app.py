import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pandas as pd
from datetime import datetime
import os

# Thêm xử lý lỗi khi khởi tạo camera
def init_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể kết nối với camera. Vui lòng kiểm tra lại.")
            return None
        return cap
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo camera: {str(e)}")
        return None

def scan_qr_code():
    st.subheader("Quét mã QR CCCD")
    
    cap = init_camera()
    if cap is None:
        return
    
    frame_placeholder = st.empty()
    stop_button = st.button("Dừng quét")
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc frame từ camera")
                break
                
            # Xử lý frame...
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded_objects = decode(frame_rgb)
            
            for obj in decoded_objects:
                # Xử lý QR code...
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(frame_rgb, [hull], True, (0, 255, 0), 2)
                
                qr_data = obj.data.decode('utf-8')
                try:
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
                except Exception as e:
                    st.error(f"Lỗi khi xử lý dữ liệu QR: {str(e)}")
            
            frame_placeholder.image(frame_rgb, channels="RGB")
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
    finally:
        cap.release()

def face_recognition_system():
    st.subheader("Nhận diện khuôn mặt")
    
    cap = init_camera()
    if cap is None:
        return
        
    frame_placeholder = st.empty()
    stop_button = st.button("Dừng nhận diện")
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc frame từ camera")
                break
                
            # Xử lý nhận diện khuôn mặt
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
          
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
    finally:
        cap.release()
