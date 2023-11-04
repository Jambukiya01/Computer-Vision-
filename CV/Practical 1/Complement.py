import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'E:\AJ Code\Paython\CV\Practical 1\OIP.jpeg'  
try:
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Failed to load the image.")
except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
    exit(1)

complement_image = 255 - image

print("Original Image Matrix:")
print(image)

print("\nComplement Image Matrix:")
print(complement_image)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Complement Image', complement_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

hist_complement = cv2.calcHist([complement_image], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_original)

plt.subplot(122)
plt.title("Complement Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_complement)

plt.tight_layout()
plt.show()
