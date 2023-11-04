import cv2
import numpy as np
import matplotlib.pyplot as plt

width, height = 400, 400
binary_image = np.ones((height, width), dtype=np.uint8) * 255

for x in range(0, width, 40):
    for y in range(0, height, 40):
        binary_image[y:y+20, x:x+20] = 0
        binary_image[y+20:y+40, x+20:x+40] = 0

cv2.imwrite("multipixel_pattern.png", binary_image)

print("Image Matrix:")
print(binary_image)

plt.figure(figsize=(8, 8))
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image with Multipixel Pattern')
plt.axis('off')
plt.show()

histogram = cv2.calcHist([binary_image], [0], None, [256], [0, 256])
plt.figure(figsize=(8, 4))
plt.plot(histogram)
plt.title('Histogram of Binary Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
