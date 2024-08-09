import cv2
import numpy as np
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
import easyocr
import threading
import time

# AprilTag detector setup
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Global variables for vital signs
pulse, spo2, dia, sys = "", "", "", ""
last_ocr_time = 0
ocr_interval = 2  # Perform OCR every 2 seconds

def sort_corners(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return corners[sorted_indices]

def process_image_with_easyocr(image, regions):
    global pulse, spo2, dia, sys
    reader = easyocr.Reader(['en'], gpu=False)
    for key, value in regions.items():
        x1, y1, x2, y2 = value
        roi = image[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (100, 50))  # Further reduce size for faster processing
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        result = reader.readtext(threshold, detail=0)
        detected_text = " ".join([text for text in result]).strip()
        
        if key == 'pulse':
            pulse = detected_text
        elif key == 'spo2':
            spo2 = detected_text
        elif key == 'Dia':
            dia = detected_text
        elif key == 'Sys':
            sys = detected_text

def ocr_thread(image, regions):
    process_image_with_easyocr(image, regions)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce frame size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Setup matplotlib
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam")
        break

    # Reduce frame size for faster processing
    frame = cv2.resize(frame, (1920, 1080))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)

    pts_src_temp = []
    for tag in tags:
        corners = tag.corners
        tag_id = tag.tag_id

        if tag_id == 0:
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            cv2.putText(frame, f'ID: {tag_id}', (int(corners[0][0]), int(corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            center_x = np.mean([corner[0] for corner in corners])
            center_y = np.mean([corner[1] for corner in corners])
            pts_src_temp.append([center_x, center_y])

    if len(pts_src_temp) >= 4:
        pts_src_array = np.array(pts_src_temp[:4], dtype='float32')
        pts_src_array = sort_corners(pts_src_array)

        pts_dst = np.array([
            [0, 0],
            [639, 0],
            [639, 479],
            [0, 479]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(pts_src_array, pts_dst)
        warped_image = cv2.warpPerspective(frame, matrix, (640, 480))

        # Define regions of interest for OCR (adjusted for 640x480 image)
        regions = {
            'pulse': [465, 125, 538, 177],
            'spo2': [460, 177, 512, 222],
            'Dia': [105, 387, 170, 428],
            'Sys': [178, 386, 226, 433]
        }

        # Draw bounding boxes for each region
        for key, value in regions.items():
            x1, y1, x2, y2 = value
            cv2.rectangle(warped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(warped_image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Perform OCR on the warped image every 2 seconds
        current_time = time.time()
        if current_time - last_ocr_time > ocr_interval:
            threading.Thread(target=ocr_thread, args=(warped_image, regions)).start()
            last_ocr_time = current_time

        ax2.clear()
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image with Detection Regions')

    ax1.clear()
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Frame')

    # Display vital signs
    fig.suptitle(f'Pulse: {pulse} | SpO2: {spo2} | Dia: {dia} | Sys: {sys}', fontsize=16)

    plt.pause(0.001)

    if plt.waitforbuttonpress(0.001):
        break

cap.release()
plt.close(fig)