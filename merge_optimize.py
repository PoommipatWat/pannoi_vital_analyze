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

# Initialize AprilTag detector with optimized settings
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,  # Increase for better performance
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Initialize OCR readers with optimized settings
easy_reader = easyocr.Reader(['en'], gpu=False, quantize=True)  # Enable quantization for better performance
pad_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, enable_mkldnn=True)  # Enable MKL-DNN for better CPU performance
keras_pipeline = keras_ocr.pipeline.Pipeline()

# Open camera (you may need to change the index or use Pi Camera module)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create matplotlib window (consider using a lighter weight GUI library for Raspberry Pi)
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))
axes = [ax1, ax2, ax3, ax4]

def sort_corners(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return corners[sorted_indices]

def process_roi(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (125, 63))  # Reduce size for better performance
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    threshold = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # EasyOCR
    try:
        easy_result = easy_reader.readtext(threshold, detail=0, batch_size=1)  # Reduce batch size
        easy_text = "".join([char for char in " ".join(easy_result) if char.isdigit()]).strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        easy_text = ""

    # PaddleOCR
    try:
        pad_result = pad_ocr.ocr(roi_resized, cls=True)
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

# Updated regions of interest for vital signs (scaled for 640x480 resolution)
roi_positions = [
    {'pulse': [465, 130, 530, 177]},
    {'spo2': [463, 175, 510, 222]},
    {'Dia': [105, 387, 170, 425]},
    {'Sys': [182, 387, 229, 323]}
]

prev_time = time.time()

# Initialize dictionaries to store vital signs
vital_signs_history = {key: {'easy': [], 'paddle': [], 'keras': []} for item in roi_positions for key in item.keys()}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            center = np.mean(corners, axis=0)
            pts_src_temp.append(center)

    # Clear previous output
    for ax in axes:
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
            [640 - 1, 0],
            [640 - 1, 480 - 1],
            [0, 480 - 1]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(pts_src_array, pts_dst)
        warped_image = cv2.warpPerspective(frame, matrix, (640, 480))

        # Process ROIs and detect vital signs
        vital_signs = {}
        for i, item in enumerate(roi_positions):
            for key, value in item.items():
                roi_image, easy_text, pad_text, keras_text = process_roi(warped_image, value[0], value[1], value[2], value[3])
                vital_signs[key] = f"E:{easy_text}, P:{pad_text}, K:{keras_text}"
                
                # Store detected values
                if len(vital_signs_history[key]['easy']) < sample_len:
                    vital_signs_history[key]['easy'].append(easy_text)
                if len(vital_signs_history[key]['paddle']) < sample_len:
                    vital_signs_history[key]['paddle'].append(pad_text)
                if len(vital_signs_history[key]['keras']) < sample_len:
                    vital_signs_history[key]['keras'].append(keras_text)
                
                # Display ROI
                axes[i].imshow(roi_image, cmap='gray')
                axes[i].set_title(f'{key}: {vital_signs[key]} ({len(vital_signs_history[key]["easy"])}/{sample_len})')
                axes[i].axis('off')

        # Display warped image
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image')
        ax2.axis('off')

        # Display detected vital signs as text
        plt.figtext(0.5, 0.02, f"Vital Signs: {vital_signs}", ha="center", fontsize=8,
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    else:
        ax2.text(0.5, 0.5, 'Waiting for 4 AprilTags', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
        for ax in axes[2:]:
            ax.text(0.5, 0.5, 'No ROI', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Increase pause time to reduce CPU usage

    # Check if all vital signs have 3 readings from all OCR systems
    # if all(len(values['easy']) >= sample_len and len(values['paddle']) >= sample_len and len(values['keras']) >= sample_len for values in vital_signs_history.values()):
    #     break

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Use OpenCV for key press detection
        break

cap.release()
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

# Print execution time in seconds
print(f'Execution time: {time.time() - prev_time:.2f} seconds')