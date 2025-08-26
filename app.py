from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Load image
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, image)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
