import os
import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from sr_predict import run_sr, load_model as load_sr_model
from detect_predict import run_fracture_detection, load_detection_model

load_dotenv()

UPLOAD_FOLDER = "images/uploads"
OUTPUT_FOLDER = "images/outputs"
DETECT_FOLDER = "images/detections"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

app = Flask(__name__)

sr_model = load_sr_model()
detect_model = load_detection_model(os.path.join(os.path.dirname(__file__), "models", "best_fracture_detection.pth"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sr_detect", methods=["GET", "POST"])
def sr_detect():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return "No file uploaded"

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(input_path)

        # Convert jfif to jpg if needed
        if filename.lower().endswith('.jfif'):
            img_temp = cv2.imread(input_path)
            if img_temp is None:
                return "Could not read image", 400
            new_filename = filename.rsplit('.', 1)[0] + '.jpg'
            new_input_path = os.path.join(UPLOAD_FOLDER, new_filename)
            cv2.imwrite(new_input_path, img_temp)
            os.remove(input_path)
            input_path = new_input_path
            filename = new_filename

        output_path = os.path.join(OUTPUT_FOLDER, filename)

        run_sr(sr_model, input_path, output_path)

        return render_template(
            "sr_detect.html",
            input_image=filename,
            output_image=filename
        )

    return render_template("sr_detect.html")

def draw_fracture_visualization(img_path, boxes_512, scores, result, confidence, output_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h, w = img.shape[:2]

    # Scale boxes
    scale_x = w / 512.0
    scale_y = h / 512.0
    boxes = np.array([
        [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
        for x1,y1,x2,y2 in boxes_512
    ])

    # Draw boxes
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), max(2, int(w*0.008)))

    # Top-left label
    if result == "Fracture":
        label = f"Fracture ({confidence:.1%})"
        color = (0, 0, 255)
        bg = (20, 20, 60)
    else:
        label = f"Non Fractured ({confidence:.1%})"
        color = (0, 255, 0)
        bg = (20, 60, 20)

    font_scale = max(0.35, w / 600.0)
    thick = 2 if w <= 50 else max(1, int(w / 300))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

    x, y = 15, 15 + th + 8
    # if w > 512:
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), bg, cv2.FILLED)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick + 1)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

    cv2.imwrite(output_path, img)

@app.route("/detect_fracture", methods=["POST"])
def detect_fracture():
    filename = request.form["filename"]
    sr_path = os.path.join(OUTPUT_FOLDER, filename)
    detect_path = os.path.join(DETECT_FOLDER, filename)

    result, conf, boxes, scores = run_fracture_detection(detect_model, sr_path, 0.45)
    draw_fracture_visualization(sr_path, boxes, scores, result, conf, detect_path)

    return render_template("final_result.html",
                           input_image=filename,
                           output_image=filename,  # only this route has it
                           detect_image=filename,
                           # result=result,
                           confidence=conf
                           )

@app.route("/detect_only", methods=["GET", "POST"])
def detect_only():
    if request.method == "GET":
        return render_template("detect_only.html")

    file = request.files.get("image")
    if not file or not file.filename:
        return "No file", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    detect_path = os.path.join(DETECT_FOLDER, filename)

    file.save(input_path)

    # Convert jfif to jpg if needed
    if filename.lower().endswith('.jfif'):
        img_temp = cv2.imread(input_path)
        if img_temp is None:
            return "Could not read image", 400
        new_filename = filename.rsplit('.', 1)[0] + '.jpg'
        new_input_path = os.path.join(UPLOAD_FOLDER, new_filename)
        cv2.imwrite(new_input_path, img_temp)
        os.remove(input_path)
        input_path = new_input_path
        filename = new_filename
        detect_path = os.path.join(DETECT_FOLDER, filename)

    result, conf, boxes, scores = run_fracture_detection(detect_model, input_path, 0.45)
    draw_fracture_visualization(input_path, boxes, scores, result, conf, detect_path)

    return render_template("final_result.html",
                           input_image=filename,
                           output_image=None,
                           detect_image=filename,
                           confidence=conf
                           )

@app.route("/uploads/<filename>")
def input_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/detections/<filename>")
def detect_file(filename):
    return send_from_directory(DETECT_FOLDER, filename)

@app.route("/contact")
def contact():
    key = os.environ.get("EMAILJS_PUBLIC_KEY")
    service = os.environ.get("EMAILJS_SERVICE_ID")
    template = os.environ.get("EMAILJS_TEMPLATE_ID")
    print(f"KEY: {key}, SERVICE: {service}, TEMPLATE: {template}")
    return render_template("contact.html",
        emailjs_public_key=key,
        emailjs_service_id=service,
        emailjs_template_id=template
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=port)

