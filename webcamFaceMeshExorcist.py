

import cv2
from face_mesh_utils import ExorcistFace

show_webcam = True
max_people = 1

# Image to swap face with
# exorcist_image_url = "https://i.pinimg.com/originals/b8/1e/07/b81e07f23c249d7f8ad0d918ac577602.jpg" #hulk
exorcist_image_url = "https://i.guim.co.uk/img/media/fbb1974c1ebbb6bf4c4beae0bb3d9cb93901953c/10_7_2380_1428/master/2380.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=223c0e9582e77253911be07c8cad564f"  # joker

# Initialize ExorcistFace class
draw_exorcist = ExorcistFace(exorcist_image_url, show_webcam, max_people)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Exorcist face", cv2.WINDOW_NORMAL)

while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    ret, exorcist_image = draw_exorcist(frame)

    if not ret:
        continue

    cv2.imshow("Exorcist face", exorcist_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# References:
# - Original model: https://google.github.io/mediapipe/solutions/face_mesh.html
# - Face swap example: https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python
