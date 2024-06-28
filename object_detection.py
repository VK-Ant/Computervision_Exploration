import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
import time

# Initialize Mediapipe drawing and objectron solutions
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

st.title("🤗VK: 3D Object Detection🤗")
with st.sidebar:
    st.title('🦜️🔗VK - 3D Object Detection (CUP,CHAIR, CAMERA) using Mediapipe🤗')
# Sidebar for selecting the webcam
webcam_id = st.sidebar.selectbox("Select Webcam", [0, 1, 2], index=2)

# Sidebar for selecting the maximum number of objects
max_num_objects = st.sidebar.selectbox("Max Number of Objects", range(1, 6), index=1)

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
    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=max_num_objects,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.8,
                                model_name='Cup') as objectron:
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

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

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
