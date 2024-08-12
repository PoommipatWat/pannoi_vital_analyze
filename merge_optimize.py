import cv2
import numpy as np
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
import easyocr
from paddleocr import PaddleOCR
import keras_ocr
from collections import Counter
import time

sample_len = 3

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

# Initialize OCR readers
easy_reader = easyocr.Reader(['en'], gpu=False)
pad_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
keras_pipeline = keras_ocr.pipeline.Pipeline()

# Open webcam
cap = cv2.VideoCapture(-1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Create matplotlib window
plt.ion()
fig = plt.figure(figsize=(20, 10))
ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
ax3 = plt.subplot2grid((2, 4), (1, 0))
ax4 = plt.subplot2grid((2, 4), (1, 1))
ax5 = plt.subplot2grid((2, 4), (1, 2))
ax6 = plt.subplot2grid((2, 4), (1, 3))
axes = [ax3, ax4, ax5, ax6]

# Add status indicator
status_ax = fig.add_axes([0.4, 0.965, 0.2, 0.03])
status_ax.axis('off')

def update_status(status, details=""):
    status_ax.clear()
    status_ax.axis('off')
    if status == "Collecting":
        color = 'blue'
    elif status == "Collection Complete":
        color = 'green'
    elif status == "Processing":
        color = 'orange'
    elif status == "Complete":
        color = 'purple'
    status_ax.text(0.5, 0.5, f'{status}{details}', ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    plt.draw()

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
    
    # EasyOCR
    try:
        easy_result = easy_reader.readtext(threshold, detail=0)
        easy_text = "".join([char for char in " ".join(easy_result) if char.isdigit()]).strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        easy_text = ""
    # PaddleOCR
    try:
        pad_result = pad_ocr.ocr(threshold, cls=True)
        pad_text = "".join([char for char in pad_result[0][0][1][0] if char.isdigit()]) if pad_result[0] else ""
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        pad_text = ""
    # Keras-OCR
    try:
        keras_result = keras_pipeline.recognize([roi_resized])
        keras_text = "".join([char for char in keras_result[0][0][0] if char.isdigit()])
    except Exception as e:
        print(f"Keras-OCR error: {e}")
        keras_text = ""
    
    return threshold, easy_text, pad_text, keras_text

# Updated regions of interest for vital signs
roi_positions = [
    {'pulse': [1372, 262, 1570, 370]},
    {'spo2': [1350, 370, 1495, 478]},
    {'Dia': [290, 838, 482, 935]},
    {'Sys': [500, 838, 645, 935]}
]

prev_time = time.time()

# Initialize list to store warped images
warped_images = []

# Collect warped images
update_status("Collecting")
while len(warped_images) < sample_len:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
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
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.clear()

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
        warped_images.append(warped_image)

        # Display warped image
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Warped Image {len(warped_images)}/{sample_len}')
        ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'Waiting for 4 AprilTags', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    if plt.waitforbuttonpress(0.001):
        break

cap.release()

update_status("Collection Complete")
plt.pause(1)  # Pause to show the "Collection Complete" status

# Process collected warped images
update_status("Processing")
vital_signs_history = {key: {'easy': [], 'paddle': [], 'keras': []} for item in roi_positions for key in item.keys()}

for idx, warped_image in enumerate(warped_images):
    update_status("Processing", f' {idx+1}/{sample_len}')
    vital_signs = {}
    for i, item in enumerate(roi_positions):
        for key, value in item.items():
            roi_image, easy_text, pad_text, keras_text = process_roi(warped_image, value[0], value[1], value[2], value[3])
            vital_signs[key] = f"Easy: {easy_text}, Paddle: {pad_text}, Keras: {keras_text}"
            
            vital_signs_history[key]['easy'].append(easy_text)
            vital_signs_history[key]['paddle'].append(pad_text)
            vital_signs_history[key]['keras'].append(keras_text)
            
            # Display ROI
            axes[i].clear()
            axes[i].imshow(roi_image, cmap='gray')
            axes[i].set_title(f'{key}: E:{easy_text} P:{pad_text} K:{keras_text}')
            axes[i].axis('off')

    # Display warped image
    ax2.clear()
    ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Processing Warped Image {idx+1}/{sample_len}')
    ax2.axis('off')

    # Display detected vital signs as text
    plt.figtext(0.5, 0.02, f"Detected Vital Signs: {vital_signs}", ha="center", fontsize=12,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)  # Pause to show each processed image

update_status("Complete")
plt.pause(1)  # Pause to show the "Complete" status

plt.close(fig)

# Calculate and display results for each vital sign
print("\nFinal Results:")
for key, values in vital_signs_history.items():
    combined_values = values['easy'] + values['paddle'] + values['keras']
    if combined_values:
        mode = Counter(combined_values).most_common(1)[0][0]
        print(f"\n{key}:")
        print(f"Mode (considered as correct value): {mode}")
        print(f"{sample_len} Samples readings")
        print("EasyOCR readings:", values['easy'])
        print("PaddleOCR readings:", values['paddle'])
        print("Keras-OCR readings:", values['keras'])
        
        # Calculate accuracy
        easy_correct = sum(1 for v in values['easy'] if v == mode)
        paddle_correct = sum(1 for v in values['paddle'] if v == mode)
        keras_correct = sum(1 for v in values['keras'] if v == mode)
        
        easy_accuracy = (easy_correct / sample_len) * 100
        paddle_accuracy = (paddle_correct / sample_len) * 100
        keras_accuracy = (keras_correct / sample_len) * 100
        
        print(f"EasyOCR Accuracy: {easy_accuracy:.2f}%")
        print(f"PaddleOCR Accuracy: {paddle_accuracy:.2f}%")
        print(f"Keras-OCR Accuracy: {keras_accuracy:.2f}%")
    else:
        print(f"{key}: No readings")

# Calculate overall accuracy
total_easy_correct = sum(sum(1 for v in values['easy'] if v == Counter(values['easy'] + values['paddle'] + values['keras']).most_common(1)[0][0]) for values in vital_signs_history.values())
total_paddle_correct = sum(sum(1 for v in values['paddle'] if v == Counter(values['easy'] + values['paddle'] + values['keras']).most_common(1)[0][0]) for values in vital_signs_history.values())
total_keras_correct = sum(sum(1 for v in values['keras'] if v == Counter(values['easy'] + values['paddle'] + values['keras']).most_common(1)[0][0]) for values in vital_signs_history.values())

total_easy_accuracy = (total_easy_correct / (sample_len * len(vital_signs_history))) * 100
total_paddle_accuracy = (total_paddle_correct / (sample_len * len(vital_signs_history))) * 100
total_keras_accuracy = (total_keras_correct / (sample_len * len(vital_signs_history))) * 100

print("\nOverall Accuracy:")
print(f"EasyOCR: {total_easy_accuracy:.2f}%")
print(f"PaddleOCR: {total_paddle_accuracy:.2f}%")
print(f"Keras-OCR: {total_keras_accuracy:.2f}%")

print(f'Total processing time: {time.time() - prev_time:.2f} seconds')