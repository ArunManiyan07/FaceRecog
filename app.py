from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import base64
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return "No file selected"

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            return process_image(filepath)

        # Handle webcam capture
        elif "webcam_image" in request.form:
            img_data = request.form["webcam_image"]
            # Remove header like 'data:image/png;base64,'
            img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Save the webcam image
            result_filename = "webcam_capture.png"
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            cv2.imwrite(result_path, image)

            return process_image(result_path)

    # First load
    return render_template("index.html", result_image=None, face_count=None)


def process_image(filepath):
    """Detect faces, draw rectangles, save processed image"""
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    result_filename = "result_" + os.path.basename(filepath)
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
    cv2.imwrite(result_path, image)

    return render_template(
        "index.html",
        result_image=result_filename,
        face_count=face_count
    )


if __name__ == "__main__":
    app.run(debug=True)
