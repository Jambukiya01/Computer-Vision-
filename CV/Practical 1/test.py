import cv2
import matplotlib.pyplot as plt

# Load an RGB image
image_path = "E:\AJ Code\Paython\CV\Practical 1\input_gray_image.jpg"  # Replace with the path to your RGB image
rgb_image = cv2.imread(image_path)

if rgb_image is None:
    print("Image not found or couldn't be loaded.")
else:
    # Display the image matrix
    print("Image Matrix:")
    print(rgb_image)

    # Separate the color channels
    blue_channel, green_channel, red_channel = cv2.split(rgb_image)

    # Plot the RGB image
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')

    # Plot the Red channel histogram
    plt.subplot(222)
    plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.6, rwidth=0.8)
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot the Green channel histogram
    plt.subplot(223)
    plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.6, rwidth=0.8)
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot the Blue channel histogram
    plt.subplot(224)
    plt.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.6, rwidth=0.8)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
