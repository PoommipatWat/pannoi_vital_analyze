import cv2
import numpy as np
from pupil_apriltags import Detector
import matplotlib.pyplot as plt

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# เปิดการเชื่อมต่อกับ webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# ตัวแปรสำหรับเก็บพิกัดของมุม AprilTag ที่มี ID ที่ต้องการ
pts_src = []

# สร้างหน้าต่าง matplotlib
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

def sort_corners(corners):
    # คำนวณศูนย์กลางของภาพ
    center = np.mean(corners, axis=0)
    
    # คำนวณมุมของจุดแต่ละจุดที่สัมพันธ์กับศูนย์กลาง
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    
    # เรียงลำดับจุดตามมุม
    sorted_indices = np.argsort(angles)
    return corners[sorted_indices]

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจาก webcam")
        break

    # แปลงภาพเป็น grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับ AprilTag
    tags = at_detector.detect(gray)

    # วาดกรอบรอบ AprilTag และเก็บพิกัด
    pts_src_temp = []  # ใช้ตัวแปรชั่วคราวในการเก็บพิกัดในรอบนี้
    for tag in tags:
        corners = tag.corners
        tag_id = tag.tag_id

        # ตรวจสอบว่า ID ของ AprilTag ตรงกับที่เราต้องการ
        if tag_id == 0:  # สมมุติว่าเราต้องการตรวจจับแท็กที่มี ID = 0
            # วาดกรอบรอบ AprilTag
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # แสดง ID ของ AprilTag
            cv2.putText(frame, f'ID: {tag_id}', (int(corners[0][0]), int(corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # คำนวณศูนย์กลางของ AprilTag
            center_x = np.mean([corner[0] for corner in corners])
            center_y = np.mean([corner[1] for corner in corners])
            center = [center_x, center_y]

            # เก็บศูนย์กลางของแท็กที่ตรวจจับได้ในตัวแปรชั่วคราว
            pts_src_temp.append(center)

    # ถ้าพบ AprilTag ที่มี ID = 0 จำนวน 4 ตัว
    if len(pts_src_temp) >= 4:
        # สมมุติว่าเรามี AprilTag 4 ตัวใน 4 มุมของกรอบ
        # การเก็บพิกัดศูนย์กลางของแท็กที่ตรวจจับได้
        pts_src_array = np.array(pts_src_temp[:4], dtype='float32')
        
        # เรียงลำดับจุดกลาง
        pts_src_array = sort_corners(pts_src_array)
        print("Points source (sorted):", pts_src_array)

        # พิกัดมุมที่ต้องการให้ตรง
        pts_dst = np.array([
            [0, 0],               # มุมซ้ายบน
            [1920 - 1, 0],        # มุมขวาบน
            [1920 - 1, 1080 - 1], # มุมขวาล่าง
            [0, 1080 - 1]         # มุมซ้ายล่าง
        ], dtype='float32')
        print("Points destination:", pts_dst)

        # คำนวณ Matrix โฮโมกราฟี
        matrix = cv2.getPerspectiveTransform(pts_src_array, pts_dst)

        # แปลงภาพ
        warped_image = cv2.warpPerspective(frame, matrix, (1920, 1080))

        

        # แสดงภาพที่แปลงแล้วด้วย matplotlib
        ax2.clear()
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image')

        # รีเซ็ต pts_src เพื่อเริ่มต้นการตรวจจับใหม่
        pts_src = pts_src_temp[:4]
    else:
        # อัพเดต pts_src ด้วยค่าใหม่ถ้าไม่ครบ 4 ตัว
        pts_src = pts_src_temp

    # แสดงภาพต้นฉบับด้วย matplotlib
    ax1.clear()
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Frame')

    plt.pause(0.001)

    # ตรวจสอบการกดปุ่ม 'q' เพื่อหยุดโปรแกรม
    if plt.waitforbuttonpress(0.001):
        break

cap.release()
plt.close(fig)
