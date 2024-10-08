import cv2
import pytesseract
import matplotlib.pyplot as plt

def process_image_with_ocr(image_path, x1, y1, x2, y2, show_threshold=False):
    """
    ประมวลผลภาพด้วย OCR ในพื้นที่ที่กำหนด
    :param image_path: พาธไปยังไฟล์ภาพ
    :param x1: พิกัด x ของมุมซ้ายบน
    :param y1: พิกัด y ของมุมซ้ายบน
    :param x2: พิกัด x ของมุมขวาล่าง
    :param y2: พิกัด y ของมุมขวาล่าง
    :param show_threshold: แสดงภาพ threshold หรือไม่
    :return: ข้อความที่ตรวจพบจาก OCR
    """
    # อ่านภาพ
    image = cv2.imread(image_path)
    # ตัดภาพตามพื้นที่ที่กำหนด
    roi = image[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (500, 250))
    # แปลงภาพเป็นโทนสีเทา
    # แปลงภาพเป็นโทนสีเทา
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    # ปรับปรุงความคมชัดของภาพ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    # ใช้ threshold เพื่อแยกตัวอักษรออกจากพื้นหลัง
    threshold = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # แสดงภาพ threshold ถ้า show_threshold เป็น True
    if show_threshold:
        plt.imshow(threshold, cmap='gray')
        plt.title(f"Threshold Image ({x1},{y1}) to ({x2},{y2})")
        plt.axis('off')
        plt.show()
    
    # ใช้ Tesseract OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(threshold, config=custom_config)
    return text.strip()

# ตัวอย่างการใช้งาน
image_path = 'vital.png'
pos = [
    {'pulse': [1600, 550, 2000, 950]},
    {'spo2': [1630, 950, 1930, 1155]},
    {'Resp': [1600, 1150, 1880, 1400]},
    {'Dia': [1500, 1415, 1665, 1530]},
    {'Sys': [1690, 1422, 1830, 1540]}
]

# image_path = 'mx400.jpg'
# pos = [
#     {'pulse': [876, 696, 922, 732]},
#     {'spo2': [874, 730, 918, 780]},
#     {'Dia': [535, 900, 600, 940]},
#     {'Sys': [602, 900, 644, 934]}
# ]

for item in pos:
    for key, value in item.items():
        detected_text = process_image_with_ocr(image_path, value[0], value[1], value[2], value[3], show_threshold=True)
        print(f"{key}: {detected_text}")