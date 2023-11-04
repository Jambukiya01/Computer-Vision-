import cv2
import numpy as np
import matplotlib.pyplot as plt


input_image_path = 'E:\AJ Code\Paython\CV\Practical 1\OIP.jpeg'  # Replace with the path to your RGB image
try:
    rgb_image = cv2.imread(input_image_path)
    if rgb_image is None:
        raise Exception("Failed to load the RGB image.")
except FileNotFoundError:
    print(f"Error: The file '{input_image_path}' was not found.")
    exit(1)


print("Original RGB Image Matrix:")
print(rgb_image)


cv2.imshow('Original RGB Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)


print("\nGrayscale Image Matrix:")
print(gray_image)


cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])


plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_gray)


rgb_from_gray = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


print("\nRGB Image Matrix from Grayscale:")
print(rgb_from_gray)


cv2.imshow('RGB Image from Grayscale', rgb_from_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


hist_rgb_from_gray = cv2.calcHist([rgb_from_gray], [0], None, [256], [0, 256])


plt.subplot(122)
plt.title("RGB Image Histogram from Grayscale")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_rgb_from_gray)


plt.tight_layout()
plt.show()
