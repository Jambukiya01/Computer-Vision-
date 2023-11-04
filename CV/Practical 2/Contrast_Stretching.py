import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image_path = 'E:\AJ Code\Paython\CV\Practical 2\_test.png'  # Replace with the path to your grayscale image
try:
    gray_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise Exception("Failed to load the grayscale image.")
except FileNotFoundError:
    print(f"Error: The file '{input_image_path}' was not found.")
    exit(1)
min_intensity = 50  # Adjust as needed
max_intensity = 200  # Adjust as needed
stretched_image = cv2.normalize(gray_image, None, min_intensity, max_intensity, cv2.NORM_MINMAX)
print("Original Image Matrix:")
print(gray_image)
print("\nStretched Image Matrix:")
print(stretched_image)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_stretched = cv2.calcHist([stretched_image], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title("Histogram before Stretching")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_original)
plt.subplot(122)
plt.title("Histogram after Stretching")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_stretched)
plt.tight_layout()
plt.show()
output_image_path = 'E:\AJ Code\Paython\CV\Practical 2\output_stretched_image.jpg'  # Replace with the desired output image file path
cv2.imwrite(output_image_path, stretched_image)
print(f"Stretched Image successfully written to '{output_image_path}'.")
