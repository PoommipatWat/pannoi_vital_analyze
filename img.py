import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# อ่านไฟล์ภาพ
img = mpimg.imread('size.png')

# แสดงภาพ
plt.imshow(img)

# ซ่อนแกน
plt.axis('off')

# แสดงหน้าต่างภาพ
plt.show()