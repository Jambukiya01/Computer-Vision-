import cv2
import numpy as np
import matplotlib.pyplot as plt
image1_path = 'E:\AJ Code\Paython\CV\Practical 5\_rose.jpg'
image2_path = 'E:\AJ Code\Paython\CV\Practical 5\OIP.jpeg'
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
if image1 is None or image2 is None:
    print("Failed to load images.")
    exit(1)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
match_distances = [m.distance for m in good_matches]
plt.hist(match_distances, bins=20, range=[0, 300], color='blue', alpha=0.7)
plt.title('Histogram of Match Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()
matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)
cv2.imshow("SIFT Matches", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
