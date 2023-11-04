import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image_path = 'E:\AJ Code\Paython\CV\Practical 4\_test.png'  # Replace with the path to your input image
try:
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("Failed to load the image.")
except FileNotFoundError:
    print(f"Error: The file '{input_image_path}' was not found.")
    exit(1)
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 0
filtered_f_transform = f_transform_shifted * mask
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_f_transform)).real
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_filtered = cv2.calcHist([filtered_image.astype(np.uint8)], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133)
plt.title("Histograms")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist_original, color='blue', label='Original')
plt.plot(hist_filtered, color='red', label='Filtered')
plt.legend()
plt.tight_layout()
plt.show()
