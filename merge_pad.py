import cv2
import numpy as np
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
import easyocr

# Initialize AprilTag detector
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Create matplotlib window
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

def sort_corners(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return corners[sorted_indices]

def process_roi(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (250, 125))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    threshold = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    result = reader.readtext(threshold, detail=0)
    detected_text = " ".join(result).strip()
    return detected_text

# Define regions of interest for vital signs
roi_positions = [
    {'pulse': [876, 696, 922, 732]},
    {'spo2': [874, 730, 918, 780]},
    {'Dia': [535, 900, 595, 940]},
    {'Sys': [602, 900, 644, 934]}
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)

    pts_src_temp = []
    for tag in tags:
        if tag.tag_id == 0:
            corners = tag.corners
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {tag.tag_id}', (int(corners[0][0]), int(corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            center = np.mean(corners, axis=0)
            pts_src_temp.append(center)

    # Clear previous output
    ax1.clear()
    ax2.clear()

    # Display original frame
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Frame')
    ax1.axis('off')

    if len(pts_src_temp) >= 4:
        pts_src_array = np.array(pts_src_temp[:4], dtype='float32')
        pts_src_array = sort_corners(pts_src_array)
        
        pts_dst = np.array([
            [0, 0],
            [1920 - 1, 0],
            [1920 - 1, 1080 - 1],
            [0, 1080 - 1]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(pts_src_array, pts_dst)
        warped_image = cv2.warpPerspective(frame, matrix, (1920, 1080))

        # Process ROIs and detect vital signs
        vital_signs = {}
        for item in roi_positions:
            for key, value in item.items():
                detected_text = process_roi(warped_image, value[0], value[1], value[2], value[3])
                vital_signs[key] = detected_text

        # Display warped image
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image')
        ax2.axis('off')

        # Display detected vital signs as text
        plt.figtext(0.5, 0.01, f"Detected Vital Signs: {vital_signs}", ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    else:
        ax2.text(0.5, 0.5, 'Waiting for 4 AprilTags', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    if plt.waitforbuttonpress(0.001):
        break

cap.release()
plt.close(fig)