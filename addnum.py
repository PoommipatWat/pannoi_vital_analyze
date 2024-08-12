import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def add_numbers_to_image(input_image_path, numbers, positions, font_path, font_sizes, text_colors, tag_image_path, background_image_path):
    # อ่านรูปภาพด้วย OpenCV
    img = cv2.imread(input_image_path)

    # แปลงรูปภาพเป็น RGB (OpenCV ใช้ BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # สร้าง PIL Image จากอาร์เรย์ NumPy
    pil_img = Image.fromarray(img_rgb)

    # สร้าง ImageDraw object
    draw = ImageDraw.Draw(pil_img)

    # ตรวจสอบให้แน่ใจว่าจำนวนตำแหน่ง ขนาดฟอนต์ และสีตรงกับจำนวนตัวเลข
    if len(numbers) != len(positions) or len(numbers) != len(font_sizes) or len(numbers) != len(text_colors):
        raise ValueError("จำนวนของตำแหน่ง ขนาดฟอนต์ สี และตัวเลขต้องเท่ากัน")

    # เพิ่มตัวเลขลงในรูปภาพ
    for number, position, font_size, text_color in zip(numbers, positions, font_sizes, text_colors):
        # โหลดฟอนต์
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, str(number), font=font, fill=text_color)

    # แปลงกลับเป็นอาร์เรย์ NumPy และ BGR
    img_with_numbers = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # แสดงรูปภาพ
    img_with_numbers = cv2.resize(img_with_numbers, (940, 525))

    # อ่านพื้นหลังจากไฟล์
    background = cv2.imread(background_image_path)
    background = cv2.resize(background, (1920, 1080))

    # คำนวณตำแหน่งที่จะวาง img_with_numbers บนพื้นหลัง
    x_offset = (background.shape[1] - img_with_numbers.shape[1]) // 2 
    y_offset = (background.shape[0] - img_with_numbers.shape[0]) // 2

    # วาง img_with_numbers บนพื้นหลัง
    background[y_offset:y_offset+img_with_numbers.shape[0], x_offset:x_offset+img_with_numbers.shape[1]] = img_with_numbers

    background = cv2.resize(background, (1280, 720))

    # บันทึกและแสดงรูปภาพ
    save = cv2.imwrite('save.jpg', background)

    cv2.imshow('Image with Numbers', background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ตัวอย่างการใช้งาน
input_image = "mx400_clear.jpg"
numbers_to_add = [123, 78, 123, 145, 80]  # ตัวเลขตัวอย่าง ปรับตามต้องการ
positions = [(1545, 150), (1530, 305), (170, 770), (165, 965), (445, 965)]  # ตำแหน่งตัวอย่าง ปรับตามต้องการ
font_sizes = [175, 175, 175, 175, 175]  # ขนาดฟอนต์ตัวอย่าง ปรับตามต้องการ
text_colors = [(24, 221, 9), (3, 235, 254), (3, 235, 254), (252, 44, 237), (252, 44, 237)]  # สีตัวอย่าง ปรับตามต้องการ (สีแดง, สีเขียว, สีน้ำเงิน, สีเหลือง)
font_path = "font/Franco-C4SA.ttf"  # แทนที่ด้วย path ของฟอนต์ที่คุณต้องการใช้
tag_image_path = "tag36h11_id_0.png"  # แทนที่ด้วย path ของ tag36_id_0.png
background_image_path = "bg.png"  # แทนที่ด้วย path ของ bg.png

add_numbers_to_image(input_image, numbers_to_add, positions, font_path, font_sizes, text_colors, tag_image_path, background_image_path)
print("รูปภาพพร้อมตัวเลขและแท็กในมุมทั้งสี่ถูกแสดงแล้ว กดปุ่มใดๆ บนหน้าต่างรูปภาพเพื่อปิด")
