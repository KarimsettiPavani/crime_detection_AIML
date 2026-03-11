from flask import Flask, render_template, request, Response
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

zones = ["Dwaraka", "MVP", "Gajuwaka", "NAD", "Gopalapatnam"]
crime_counts = [120, 80, 150, 60, 90]

video_path_global = None


def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.putText(frame,
                    "Suspicious Activity Detected!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    global video_path_global
    if video_path_global is None:
        return "",204
    return Response(generate_frames(video_path_global),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path_global
    if request.method == 'POST':
        file = request.files.get('video')

        if file and file.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(video_path)
            video_path_global = video_path

            return render_template(
                "index.html",
                zones=zones,
                crime_counts=crime_counts,
                show_video=True,
                active_section="cctv"
            )

    # This ALWAYS runs if:
    # - GET request
    # - POST but no file
    # So Flask always returns something

    return render_template(
        "index.html",
        zones=zones,
        crime_counts=crime_counts,
        show_video=False,
        active_section="risk"
    )

if __name__ == "__main__":
    app.run(debug=True)