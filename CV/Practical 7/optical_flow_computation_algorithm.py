import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('E:\AJ Code\Paython\CV\Practical 7\_nat.mp4')  # Replace with the path to your video file
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev_frame)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
flow_magnitudes = []
flow_directions = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prevPts, None)
    flow_magnitude = np.linalg.norm(nextPts - prevPts, axis=2)
    flow_direction = np.arctan2(nextPts[..., 1] - prevPts[..., 1], nextPts[..., 0] - prevPts[..., 0])
    flow_magnitudes.append(flow_magnitude)
    flow_directions.append(flow_direction)
    for i, (new, old) in enumerate(zip(nextPts, prevPts)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    result = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', result)
    prev_gray = gray.copy()
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        break
cap.release()
cv2.destroyAllWindows()
flow_magnitudes = np.concatenate(flow_magnitudes)
flow_directions = np.concatenate(flow_directions)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.hist(flow_magnitudes, bins=20, range=(0, np.max(flow_magnitudes)), color='blue', alpha=0.7)
plt.title('Histogram of Flow Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(flow_directions, bins=20, range=(-np.pi, np.pi), color='green', alpha=0.7)
plt.title('Histogram of Flow Directions')
plt.xlabel('Direction (radians)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
