import cv2
import numpy as np
image_path = 'E:\AJ Code\Paython\CV\Practical 10\_4_face.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
classifier_path = 'E:\AJ Code\Paython\CV\Practical 10\haarcascade_frontalcatface.xml'
detector = cv2.CascadeClassifier(classifier_path)
objects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in objects:
    detected_object = image[y:y + h, x:x + w]
    object_hist = cv2.calcHist([detected_object], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    hist_image = np.zeros((256, 256, 3), np.uint8)
    hist_image[:, :, 0] = object_hist[:, :, 0]
    hist_image[:, :, 1] = object_hist[:, :, 1]
    hist_image[:, :, 2] = object_hist[:, :, 2]
    hist_image = cv2.resize(hist_image, (200, 200))
    cv2.imwrite('detected_object.jpg', detected_object)
    cv2.imwrite('object_histogram.jpg', hist_image)
for (x, y, w, h) in objects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Object Detection and Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
