import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
image_path = "E:\AJ Code\Paython\CV\Practical 1\_asd.jpg"
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if binary_image is None:
    print("Image not found or couldn't be loaded.")
else:
    # Display the image matrix
    print("Image Matrix:")
    print(binary_image)

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    plt.show()

    # Compute and plot the histogram
    histogram = cv2.calcHist([binary_image], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 4))
    plt.plot(histogram)
    plt.title('Histogram of Binary Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
