import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
input_image_path = 'E:\AJ Code\Paython\CV\Practical 2\_test.png'  # Replace with the path to your grayscale image

try:
    gray_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise Exception("Failed to load the grayscale image.")
except FileNotFoundError:
    print(f"Error: The file '{input_image_path}' was not found.")
    exit(1)

# Display the original image matrix
print("Original Image Matrix:")
print(gray_image)

# Perform histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display the equalized image matrix
print("\nEqualized Image Matrix:")
print(equalized_image)

# Calculate histograms
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# Display original and equalized histograms
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title("Original Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_original)

plt.subplot(122)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_equalized)

plt.tight_layout()
plt.show()

# Save the equalized image
output_image_path = 'your_output_equalized_image.jpg'  # Replace with the desired output image file path
cv2.imwrite(output_image_path, equalized_image)
print(f"Equalized Image successfully written to '{output_image_path}'.")
