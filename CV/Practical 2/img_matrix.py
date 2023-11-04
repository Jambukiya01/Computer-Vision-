import cv2

# Load an image
image_path = "E:\AJ Code\Paython\CV\Practical 2\_test.png"  # Replace with the path to your image
image = cv2.imread(image_path)

if image is None:
    print("Image not found or couldn't be loaded.")
else:
    # Display the image matrix
    print("Image Matrix:")
    print(image)
