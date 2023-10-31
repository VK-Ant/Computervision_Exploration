import cv2
import mediapipe as mp
import numpy as np
from imread_from_url import imread_from_url

# Initialize webcam
cap = cv2.VideoCapture(1)
cv2.namedWindow("Halloween Background", cv2.WINDOW_NORMAL)

# Read background image
background_image_url = "https://cdn.pixabay.com/photo/2015/09/26/13/25/halloween-959047_960_720.jpg"
background_image = imread_from_url(background_image_url)

# Read smoke image
smoke_image_url = "https://images.unsplash.com/photo-1542789828-6c82d889ed74?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&dl=eberhard-grossgasteiger-HlJZ-xm3KCI-unsplash.jpg"
smoke_image = imread_from_url(smoke_image_url)

# Resize the background image to the resolution of the webcam
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
background_image = cv2.resize(background_image, (webcam_width, webcam_height), interpolation = cv2.INTER_AREA)
smoke_image = cv2.resize(smoke_image, (webcam_width, webcam_height), interpolation = cv2.INTER_AREA)

# Inialize background segmentation (0: default model, 1: landmark image optimized)
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(1)


from pumpkin_face_utils import read_pumpkin_image, draw_pumpkins

show_webcam = False

# Initialize webcam
cap = cv2.VideoCapture(1)
cv2.namedWindow("Pumpkin face", cv2.WINDOW_NORMAL)

# Read pumpkin image
pumpkin_image_path = "https://cdn.pixabay.com/photo/2017/10/01/11/36/pumpkin-2805140_960_720.png"
pumpkin_image = read_pumpkin_image(pumpkin_image_path)

# Inialize background segmentation (0: small model for distace < 2m, 1: full range model for distance < 5m)
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	if not ret:
		continue

	# Flip the image horizontally
	frame = cv2.flip(frame, 1)

	# Extract background
	input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	input_image.flags.writeable = False
	background_mask = cv2.cvtColor(selfie_segmentation.process(input_image).segmentation_mask,cv2.COLOR_GRAY2RGB)

	# Combine the webcam image with smoke
	smoke_frame = cv2.addWeighted(frame, 0.4, smoke_image, 0.6, 0)

	# Fill the background mask with the background image (Multiply with the mask to get a smoother combination)
	combined_image = np.uint8(smoke_frame * background_mask + background_image * (1-background_mask))

	img_height, img_width, _ = frame.shape


	# Flip the image horizontally
	frame1 = cv2.flip(frame, 1)

	# Detect face
	input_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
	input_image.flags.writeable = False
	detections = face_detection.process(input_image).detections

	# Draw pumkins
	output_img = draw_pumpkins(frame1, pumpkin_image, detections, show_webcam)

	cv2.imshow("Pumpkin face", output_img)


	cv2.imshow('Halloween Background', combined_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break