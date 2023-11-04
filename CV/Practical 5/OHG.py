import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
image_path = 'E:\AJ Code\Paython\CV\Practical 5\OIP.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Failed to load the image '{image_path}'.")
    exit(1)
fd, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hist, bin_edges = np.histogram(fd, bins=9, range=(0, 180), density=True)
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133)
plt.bar(bin_edges[:-1], hist, width=20)
plt.title('HOG Histogram'), plt.xlabel('Gradient Orientation'), plt.ylabel('Normalized Frequency')
plt.tight_layout()
plt.show()
