import cv2
from tracker import *

cap = cv2.VideoCapture('highway.mp4')

tracker = EuclideanDistTracker()

# success, frame = cap.read()
# roi = cv2.selectROI(frame)
# print(roi)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    success, frame = cap.read()

    if not success:
        break

    roi = frame[243:500, 596:767]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append([x,y,w,h])
            # cv2.drawContours(roi, contour, -1, (0,255,0), 2)

    bbox_ids = tracker.update(detections)
    for bbox_id in bbox_ids:
        x,y,w,h,id = bbox_id
        cv2.putText(roi, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    if not success:
        break

    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()