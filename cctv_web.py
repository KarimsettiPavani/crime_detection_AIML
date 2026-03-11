import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

CROWD_THRESHOLD = 8
ROBBERY_MOTION_THRESHOLD = 60000
ALERT_COOLDOWN = 3

def process_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    last_alert_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        motion_pixels = cv2.countNonZero(fgmask)

        results = model(frame, conf=0.35)
        annotated_frame = results[0].plot()

        person_count = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                person_count += 1

        cv2.putText(annotated_frame,
                    f"Persons: {person_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2)

        current_time = time.time()

        if (person_count >= 2 and
            motion_pixels > ROBBERY_MOTION_THRESHOLD and
            current_time - last_alert_time > ALERT_COOLDOWN):

            cv2.putText(annotated_frame,
                        "!!! ROBBERY ALERT !!!",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

            last_alert_time = current_time

        out.write(annotated_frame)

    cap.release()
    out.release()