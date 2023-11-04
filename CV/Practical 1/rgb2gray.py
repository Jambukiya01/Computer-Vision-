import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = "E:\AJ Code\Paython\CV\Practical 1\input_rgb_image.jpg"  # Replace with the path to your RGB image
rgb_image = cv2.imread(image_path)
if rgb_image is None:
    print("Image not found or couldn't be loaded.")
else:
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    print("Image Matrix:")
    print(gray_image)
    plt.figure(figsize=(8, 8))
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 4))
    plt.plot(histogram)
    plt.title('Histogram of Grayscale Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
