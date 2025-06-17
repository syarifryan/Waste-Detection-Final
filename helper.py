from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading
import os
from PIL import Image
import numpy as np
import ast

def sleep_and_clear_success():
    time.sleep(3)
    if 'recyclable_placeholder' in st.session_state:
        st.session_state['recyclable_placeholder'].empty()
    if 'non_recyclable_placeholder' in st.session_state:
        st.session_state['non_recyclable_placeholder'].empty()
    if 'hazardous_placeholder' in st.session_state:
        st.session_state['hazardous_placeholder'].empty()

def load_model(model_path):
    model = YOLO(model_path)
    return model

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ").title()

def save_detection_result(result_type, items, filename=None, image=None, video_path=None):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if filename is None:
        filename = f"detection_{timestamp}"
    
    result = {
        "timestamp": timestamp,
        "type": result_type,
        "items": list(items),
        "filename": filename
    }
    
    # Save the detection image if provided
    if image is not None:
        img_path = os.path.join(st.session_state['results_dir'], f"{filename}.jpg")
        try:
            cv2.imwrite(img_path, image)
            result['image_path'] = img_path
        except Exception as e:
            st.error(f"Error saving image: {e}")
    
    # Save the detection video if provided
    if video_path is not None:
        result['video_path'] = video_path
    
    # Save the result metadata
    result_path = os.path.join(st.session_state['results_dir'], f"{filename}.txt")
    try:
        with open(result_path, 'w') as f:
            for key, value in result.items():
                if key not in ['image_path', 'video_path']:
                    f.write(f"{key}: {value}\n")
    except Exception as e:
        st.error(f"Error saving result: {e}")
    
    return result

def _display_detected_frames(model, st_frame, image, save_result=False, filename=None):
    image = cv2.resize(image, (640, int(640*(9/16))))
    
    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    if 'recyclable_placeholder' not in st.session_state:
        st.session_state['recyclable_placeholder'] = st.sidebar.empty()
    if 'non_recyclable_placeholder' not in st.session_state:
        st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
    if 'hazardous_placeholder' not in st.session_state:
        st.session_state['hazardous_placeholder'] = st.sidebar.empty()

    if 'last_detection_time' not in st.session_state:
        st.session_state['last_detection_time'] = 0

    res = model.predict(image, conf=0.6)
    names = model.names
    detected_items = set()

    for result in res:
        new_classes = set([names[int(c)] for c in result.boxes.cls])
        if new_classes != st.session_state['unique_classes']:
            st.session_state['unique_classes'] = new_classes
            st.session_state['recyclable_placeholder'].markdown('')
            st.session_state['non_recyclable_placeholder'].markdown('')
            st.session_state['hazardous_placeholder'].markdown('')
            detected_items.update(st.session_state['unique_classes'])

            recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

            if recyclable_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in recyclable_items)
                st.session_state['recyclable_placeholder'].markdown(
                    f"<div class='stRecyclable'>Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            if non_recyclable_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in non_recyclable_items)
                st.session_state['non_recyclable_placeholder'].markdown(
                    f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            if hazardous_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in hazardous_items)
                st.session_state['hazardous_placeholder'].markdown(
                    f"<div class='stHazardous'>Hazardous items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )

            # Save the detection result if requested
            if save_result and detected_items:
                save_detection_result(
                    "image" if not hasattr(st.session_state, 'is_video') else "video",
                    detected_items,
                    filename,
                    image if not hasattr(st.session_state, 'is_video') else None,
                    st.session_state.get('video_path')
                )

            threading.Thread(target=sleep_and_clear_success).start()
            st.session_state['last_detection_time'] = time.time()

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR", use_container_width=True)
    return res_plotted

def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH
    if st.button('Start Detection'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            stop_button = st.button('Stop Detection')
            
            while (vid_cap.isOpened()) and not stop_button:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                else:
                    vid_cap.release()
                    break
            vid_cap.release()
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def upload_image(model, results_dir):
    st.session_state['results_dir'] = results_dir
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button('Detect Waste'):
                st.text("Detection Results...")
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                col1, col2 = st.columns(2)
                with col1:
                    st_frame = st.empty()
                
                detected_image = _display_detected_frames(
                    model, 
                    st_frame, 
                    image_cv, 
                    save_result=True,
                    filename=uploaded_file.name.split('.')[0]
                )
                
                with col2:
                    st.image(detected_image, caption='Detected Waste', channels="BGR", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

def upload_video(model, results_dir):
    st.session_state['results_dir'] = results_dir
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded video temporarily
            temp_video_path = os.path.join(results_dir, uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state['is_video'] = True
            st.session_state['video_path'] = temp_video_path
            
            st.video(uploaded_file)
            
            if st.button('Detect Waste in Video'):
                st.text("Processing Video...")
                stop_button = st.button('Stop Processing')
                vid_cap = cv2.VideoCapture(temp_video_path)
                st_frame = st.empty()
                
                while (vid_cap.isOpened()) and not stop_button:
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(model, st_frame, image, save_result=True)
                    else:
                        vid_cap.release()
                        break
                
                vid_cap.release()
                del st.session_state['is_video']
                del st.session_state['video_path']
        except Exception as e:
            st.error(f"Error processing video: {e}")
            if 'is_video' in st.session_state:
                del st.session_state['is_video']
            if 'video_path' in st.session_state:
                del st.session_state['video_path']

def display_previous_results(results_dir):
    try:
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
        
        if not result_files:
            st.write("No previous detection results found.")
            return
        
        for result_file in sorted(result_files, reverse=True)[:10]:  # Show latest 10 results
            result_path = os.path.join(results_dir, result_file)
            try:
                with open(result_path, 'r') as f:
                    result_data = {}
                    for line in f:
                        if ': ' in line:
                            key, value = line.strip().split(': ', 1)
                            result_data[key] = value
                
                with st.container():
                    st.subheader(f"Detection on {result_data.get('timestamp', 'unknown time')}")
                    cols = st.columns([1, 3])
                    
                    with cols[0]:
                        if 'image_path' in result_data and os.path.exists(result_data['image_path']):
                            st.image(result_data['image_path'], use_container_width=True)
                        elif 'video_path' in result_data and os.path.exists(result_data['video_path']):
                            st.video(result_data['video_path'])
                    
                    with cols[1]:
                        st.write(f"**Detection Type:** {result_data.get('type', 'unknown')}")
                        st.write(f"**Filename:** {result_data.get('filename', 'unknown')}")
                        
                        try:
                            items = ast.literal_eval(result_data.get('items', '[]'))
                            if items:
                                recyclable, non_recyclable, hazardous = classify_waste_type(items)
                                
                                if recyclable:
                                    detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in recyclable)
                                    st.markdown(
                                        f"<div class='stRecyclable'>Recyclable items:\n\n- {detected_items_str}</div>",
                                        unsafe_allow_html=True
                                    )
                                if non_recyclable:
                                    detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in non_recyclable)
                                    st.markdown(
                                        f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {detected_items_str}</div>",
                                        unsafe_allow_html=True
                                    )
                                if hazardous:
                                    detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in hazardous)
                                    st.markdown(
                                        f"<div class='stHazardous'>Hazardous items:\n\n- {detected_items_str}</div>",
                                        unsafe_allow_html=True
                                    )
                        except Exception as e:
                            st.error(f"Error parsing items: {e}")
                    
                    st.divider()
            except Exception as e:
                st.error(f"Error loading result {result_file}: {e}")
    except Exception as e:
        st.error(f"Error accessing results directory: {e}")