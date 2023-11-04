import cv2
import numpy as np
def main():
    capture = cv2.VideoCapture('E:\AJ Code\Paython\CV\Practical 8\_nat.mp4')  
    if not capture.isOpened():
        print("Unable to open file!")
        return
    colors = np.random.randint(0, 255, (100, 3))
    ret, old_frame = capture.read()
    if not ret:
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    mask = np.zeros_like(old_frame)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)
            img = cv2.add(frame, mask)
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_image = np.zeros((100, 256), dtype=np.uint8)
            cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
            for i in range(1, 256):
                cv2.line(hist_image, (i - 1, 100), (i, 100 - int(hist[i][0])), 255, 1)
            hist_image = cv2.cvtColor(hist_image, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Histogram', hist_image)
            cv2.imshow('Optical Flow Motion Detection', img)
        keyboard = cv2.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break


        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if p1 is not None else None

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
