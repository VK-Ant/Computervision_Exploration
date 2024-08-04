import cv2
import mediapipe as mp
import numpy as np
import time
import utils_eye

# Constants and variables
CLOSED_EYES_FRAME = 3
FONTS = cv2.FONT_HERSHEY_COMPLEX
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Mediapipe initializations
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Camera setup
camera = cv2.VideoCapture(2)

# Utility functions
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = np.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eye
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

# Main loop
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
start_time = time.time()

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        # Eye tracking
        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
        utils_eye.colorBackgroundText(frame, f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100), 2, utils_eye.BLUE, utils_eye.YELLOW)

        if ratio > 5.5:
            CEF_COUNTER += 1
            utils_eye.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_width/2), 100), 2, utils_eye.PURPLE, pad_x=6, pad_y=6)
        else:
            if CEF_COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                CEF_COUNTER = 0
        utils_eye.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150), 2)

        cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils_eye.RED, 3, cv2.LINE_AA)
        cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils_eye.RED, 3, cv2.LINE_AA)

        # Head pose estimation
        face_3d = []
        face_2d = []

        for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * frame_width, lm.y * frame_height)
                    nose_3d = (lm.x * frame_width, lm.y * frame_height, lm.z * 3000)

                x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])       

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * frame_width
        cam_matrix = np.array([[focal_length, 0, frame_height / 2],
                               [0, focal_length, frame_width / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        
        cv2.line(frame, p1, p2, (255, 0, 0), 3)
        cv2.putText(frame, f'{text}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.putText(frame, f"x: {np.round(x,2)}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"y: {np.round(y,2)}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"z: {np.round(z,2)}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    # Calculate and display FPS
    frame_counter += 1
    end_time = time.time()
    fps = frame_counter / (end_time - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),3)
    cv2.imshow('Eye Tracking and Head Pose Estimation', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()