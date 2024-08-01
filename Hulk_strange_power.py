import cv2
import mediapipe as mp
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cv2.namedWindow("Combined Stream", cv2.WINDOW_NORMAL)

# Mediapipe setup for hand landmarks
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Inner and outer circle images
INNER_CIRCLE = "Models/inner_circles/blue.png"
OUTER_CIRCLE = "Models/outer_circles/orange.png"
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

    cv2.imshow("Funny! Hulk with doctor strange power", exorcist_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
