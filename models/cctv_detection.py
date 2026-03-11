import cv2
import numpy as np
import winsound
import threading
import time
from ultralytics import YOLO

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO("yolov8s.pt")

# ----------------------------
# LOAD VIDEO
# ----------------------------
cap = cv2.VideoCapture(r"C:\Users\user\Downloads\cctv1.mp4")

# ----------------------------
# SETTINGS
# ----------------------------
CROWD_THRESHOLD = 8
ROBBERY_MOTION_THRESHOLD = 60000   # Adjust if needed
ALERT_COOLDOWN = 3  # seconds

last_alert_time = 0

# Background subtractor (stable motion detection)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# ----------------------------
# ALARM FUNCTION (Non-blocking)
# ----------------------------
def alarm_sound():
    winsound.Beep(1200, 500)

# ----------------------------
# MAIN LOOP
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 600))

    # ----------------------------
    # MOTION DETECTION
    # ----------------------------
    fgmask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    motion_pixels = cv2.countNonZero(fgmask)
    cv2.putText(annotated_frame,
            f"Motion Pixels: {motion_pixels}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2)

    # ----------------------------
    # OBJECT DETECTION
    # ----------------------------
    results = model(frame, conf=0.35)
    annotated_frame = results[0].plot()

    person_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "person":
            person_count += 1

    # ----------------------------
    # DISPLAY PERSON COUNT
    # ----------------------------
    cv2.putText(annotated_frame,
                f"Persons: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2)

    # ----------------------------
    # CROWD ALERT
    # ----------------------------
    if person_count >= CROWD_THRESHOLD:
        cv2.putText(annotated_frame,
                    "HIGH CROWD ALERT!",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # ----------------------------
    # ROBBERY / SUSPICIOUS ACTIVITY ALERT
    # ----------------------------
    current_time = time.time()

    if (person_count >= 2 and
        motion_pixels > ROBBERY_MOTION_THRESHOLD and
        current_time - last_alert_time > ALERT_COOLDOWN):

        cv2.putText(annotated_frame,
                    "!!! ROBBERY ALERT !!!",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

        threading.Thread(target=alarm_sound).start()
        last_alert_time = current_time

    # ----------------------------
    # SHOW WINDOW
    # ----------------------------
    cv2.imshow("AI CCTV Surveillance System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# RELEASE
# ----------------------------
cap.release()
cv2.destroyAllWindows()