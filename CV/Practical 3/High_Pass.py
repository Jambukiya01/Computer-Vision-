import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image_path = 'E:\AJ Code\Paython\CV\Practical 3\_rose.jpg'  
try:
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        raise Exception("Failed to load the image.")
except FileNotFoundError:
    print(f"Error: The file '{input_image_path}' was not found.")
    exit(1)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
filtered_image = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
filtered_image = cv2.convertScaleAbs(filtered_image)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_filtered = cv2.calcHist([filtered_image], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title("Histogram before Filtering")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_original)
plt.subplot(122)
plt.title("Histogram after Filtering")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_filtered)
plt.tight_layout()
plt.show()
cv2.imshow('Original Image', gray_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
