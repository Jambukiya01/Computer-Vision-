import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'E:\AJ Code\Paython\CV\Practical 6\OIP.jpeg'
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load the image '{image_path}'.")
    exit(1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pixel_values_before = image.reshape((-1, 3))
pixel_values_before = np.float32(pixel_values_before)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  
_, labels, centers = cv2.kmeans(pixel_values_before, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_segmented = cv2.calcHist([segmented_image], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.axis('off')
plt.subplot(132)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image'), plt.axis('off')
plt.subplot(133)
plt.plot(hist_original, color='blue', label='Original')
plt.plot(hist_segmented, color='red', label='Segmented')
plt.title('Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
