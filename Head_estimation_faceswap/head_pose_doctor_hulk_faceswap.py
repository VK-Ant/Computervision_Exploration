import cv2
import mediapipe as mp
import numpy as np
import time
from face_mesh_utils import ExorcistFace
from functions_hulk_strange import position_data, calculate_distance, draw_line, asd

# Image to swap face with
exorcist_image_url = "https://i.pinimg.com/originals/b8/1e/07/b81e07f23c249d7f8ad0d918ac577602.jpg"  # hulk

# Initialize ExorcistFace class
show_webcam = True
max_people = 1
draw_exorcist = ExorcistFace(exorcist_image_url, show_webcam, max_people)

# Initialize webcam
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mediapipe setup for hand landmarks
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Mediapipe setup for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Inner and outer circle images
INNER_CIRCLE = "light_orange_inner.png"
OUTER_CIRCLE = "dark_red_outer.png"

inner_circle = cv2.imread(INNER_CIRCLE, -1)
outer_circle = cv2.imread(OUTER_CIRCLE, -1)

LINE_COLOR = (0, 140, 255)
deg = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Process face swapping
    ret, exorcist_image = draw_exorcist(frame)
    if not ret:
        continue

    #cv2.putText(exorcist_image, "Face Swap: Hulk with Doctor Strange Power", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame to RGB for Mediapipe
    rgbFrame = cv2.cvtColor(exorcist_image, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgbFrame)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lmLists = []
            for id, lm in enumerate(hand.landmark):
                h, w, _ = exorcist_image.shape
                lmLists.append([int(lm.x * w), int(lm.y * h)])

            coordinates = position_data(lmLists)
            wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip = coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4], coordinates[5], coordinates[6], coordinates[7]
            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinks_distance = calculate_distance(index_tip, pinky_tip)
            ratio = index_pinks_distance / index_wrist_distance

            if 1.3 > ratio > 0.5:
                exorcist_image = draw_line(exorcist_image, wrist, thumb_tip)
                exorcist_image = draw_line(exorcist_image, wrist, index_tip)
                exorcist_image = draw_line(exorcist_image, wrist, midle_tip)
                exorcist_image = draw_line(exorcist_image, wrist, ring_tip)
                exorcist_image = draw_line(exorcist_image, wrist, pinky_tip)
                exorcist_image = draw_line(exorcist_image, thumb_tip, index_tip)
                exorcist_image = draw_line(exorcist_image, thumb_tip, midle_tip)
                exorcist_image = draw_line(exorcist_image, thumb_tip, ring_tip)
                exorcist_image = draw_line(exorcist_image, thumb_tip, pinky_tip)
            elif ratio > 1.3:
                centerx = midle_mcp[0]
                centery = midle_mcp[1]
                shield_size = 3.0
                diameter = round(index_wrist_distance * shield_size)
                x1 = round(centerx - (diameter / 2))
                y1 = round(centery - (diameter / 2))
                h, w, _ = exorcist_image.shape
                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w
                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h
                if x1 + diameter > w:
                    diameter = w - x1
                if y1 + diameter > h:
                    diameter = h - y1
                shield_size = diameter, diameter
                ang_vel = 2.0
                deg = deg + ang_vel
                if deg > 360:
                    deg = 0
                hei, wid, _ = outer_circle.shape
                cen = (wid // 2, hei // 2)
                M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0)
                M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                rotated1 = cv2.warpAffine(outer_circle, M1, (wid, hei))
                rotated2 = cv2.warpAffine(inner_circle, M2, (wid, hei))
                if diameter != 0:
                    exorcist_image = asd(rotated1, exorcist_image, x1, y1, shield_size)
                    exorcist_image = asd(rotated2, exorcist_image, x1, y1, shield_size)

    # Process head pose estimation
    start = time.time()
    head_pose_image = frame.copy()
    
    # Convert the color space from BGR to RGB
    head_pose_image = cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB)

    # To improve performance
    head_pose_image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(head_pose_image)
    
    # To improve performance
    head_pose_image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    head_pose_image = cv2.cvtColor(head_pose_image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = head_pose_image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

            # Draw face landmarks with red connections
            mp_drawing.draw_landmarks(
                image=head_pose_image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=0.5))

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          
            # See where the user's head tilting
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

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(head_pose_image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(head_pose_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            #cv2.putText(head_pose_image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            #cv2.putText(head_pose_image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            #cv2.putText(head_pose_image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    #print("FPS: ", fps)

    cv2.putText(exorcist_image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Add the text on the image
    cv2.putText(exorcist_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(exorcist_image, "x: " + str(np.round(x,2)), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(exorcist_image, "y: " + str(np.round(y,2)), (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(exorcist_image, "z: " + str(np.round(z,2)), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine both frames horizontally
    combined_image = np.hstack((head_pose_image, exorcist_image))

    cv2.imshow("VK: Head pose estimation & Faceswap (Hulk & doctorstrange)", combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
