import cv2
import numpy as np

# Load the image
image = cv2.imread('mx400.jpg')

# Define the source points (the corners of the rectangle you want to transform)
src_points = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])

# Define the destination points (where you want the source points to map to)
dst_points = np.float32([[10, 100], [300, 50], [100, 250], [300, 300]])

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

# Save or display the result
cv2.imwrite('output.jpg', warped_image)
# cv2.imshow('Warped Image', warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
