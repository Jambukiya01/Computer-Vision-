import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = "E:\AJ Code\Paython\CV\Practical 1\input_gray_image.jpg"  # Replace with the path to your grayscale image
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    print("Image not found or couldn't be loaded.")
else:
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    print("Image Matrix:")
    print(rgb_image)
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')
    plt.subplot(222)
    plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.6, rwidth=0.8)
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.subplot(223)
    plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.6, rwidth=0.8)
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.subplot(224)
    plt.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.6, rwidth=0.8)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
