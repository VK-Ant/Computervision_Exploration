'''
Author: VK
Date: 04/07/2024
Time:00:31 IST
credit: CVZONE
'''



import cv2
import cvzone
import numpy as np
import streamlit as st
import mediapipe as mp
import time
import os

st.title("😎🥸VK👓Virtual Glass Tryon😎🥸")
with st.sidebar:
    st.title('🔗VK 👓 VIRTUAL GLASS TRYON😎')

# Function to resize images
def resize_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        image = cv2.resize(image, target_size)
    return image

# Initialize Mediapipe drawing and objectron solutions
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Sidebar for selecting the webcam
webcam_id = st.sidebar.selectbox("Select Webcam", [0, 1, 2], index=2)

# Sidebar for selecting the maximum number of objects
max_num_objects = st.sidebar.selectbox("Max Number of Objects", range(1, 6), index=1)

# Directory containing glasses images
glasses_dir = "Glasses"

# Function to fetch glasses options
def fetch_glasses_options():
    glasses_options = []
    for file_name in os.listdir(glasses_dir):
        if file_name.endswith(".png"):
            image_path = os.path.join(glasses_dir, file_name)
            glasses_image = resize_image(image_path, target_size=(256, 256))
            glasses_options.append((os.path.splitext(file_name)[0], glasses_image))
    return glasses_options

# Dropdown for glasses selection
glasses_options = fetch_glasses_options()
if "selected_glasses" not in st.session_state:
    st.session_state.selected_glasses = glasses_options[0][0]

selected_glasses_index = st.sidebar.selectbox("Select Glasses", [name for name, _ in glasses_options], index=0)
selected_glasses_filename = selected_glasses_index + ".png"
st.session_state.selected_glasses = selected_glasses_filename

# Display the selected glasses image in the sidebar
selected_glasses_image = next(img for name, img in glasses_options if name == selected_glasses_index)
st.sidebar.image(selected_glasses_image, caption=f"Selected Glasses: {selected_glasses_index}")

# Start and stop buttons
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

def start_camera():
    st.session_state.run_camera = True

def stop_camera():
    st.session_state.run_camera = False

st.sidebar.button("Start Camera", on_click=start_camera)
st.sidebar.button("Stop Camera", on_click=stop_camera)

# Create a container for the video frames
frame_container = st.empty()

# Function to process the video stream
def process_video():
    cap = cv2.VideoCapture(webcam_id)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=max_num_objects,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.8,
                                model_name='Cup') as objectron:
        count = 0
        num = 1
        
        while cap.isOpened() and st.session_state.run_camera:
            success, image = cap.read()
            if not success:
                st.error("Ignoring empty camera frame.")
                continue

            start = time.time()

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = objectron.process(image)

            # Draw the box landmarks on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(image, detected_object.rotation,
                                         detected_object.translation)

            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                roi_gray = gray_scale[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                overlay_name = st.session_state.selected_glasses
                overlay_path = os.path.join(glasses_dir, overlay_name)
                overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                overlay_resize = cv2.resize(overlay, (w, int(h*0.8)))
                image = cvzone.overlayPNG(image, overlay_resize, [x, y])


            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Convert the BGR image to RGB for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            frame_container.image(image_rgb, channels="RGB")

            # Add a small delay to avoid overloading the app
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the camera processing function if the camera is running
if st.session_state.run_camera:
    process_video()
