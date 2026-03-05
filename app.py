import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from sr_predict import run_sr, load_model as load_sr_model
from detect_predict import run_fracture_detection, load_detection_model

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
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        file.save(input_path)
        run_sr(sr_model, input_path, output_path)

        return render_template(
            "sr_detect.html",
            input_image=filename,
            output_image=filename
        )

    return render_template("sr_detect.html")

# @app.route("/detect_fracture", methods=["POST"])
# def detect_fracture():
#     filename = request.form["filename"]
#     input_path = os.path.join(UPLOAD_FOLDER, filename)
#     sr_path = os.path.join(OUTPUT_FOLDER, filename)
#     detect_path = os.path.join(DETECT_FOLDER, filename)
#
#     # result, confidence = run_fracture_detection(detect_model, sr_path)
#     result, confidence, boxes, scores = run_fracture_detection(detect_model, sr_path)
#
#     # Create detection image with overlay
#     img = cv2.imread(sr_path)
#     color = (0, 255, 0) if "non" in result else (0, 0, 255)
#     cv2.putText(img, f"{result} ({confidence:.2%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#     cv2.imwrite(detect_path, img)
#
#     return render_template(
#         "final_result.html",
#         input_image=filename,
#         output_image=filename,
#         detect_image=filename
#     )

# @app.route("/detect_fracture", methods=["POST"])
# def detect_fracture():
#     filename = request.form["filename"]
#     sr_path    = os.path.join(OUTPUT_FOLDER, filename)
#     detect_path = os.path.join(DETECT_FOLDER, filename)
#
#     # Load SR image
#     img = cv2.imread(sr_path)
#     if img is None:
#         print(f"Error: Could not load SR image: {sr_path}")
#         return "Error loading super-resolved image", 500
#
#     orig_h, orig_w = img.shape[:2]
#
#     # Run detection
#     result, confidence, boxes_512, scores = run_fracture_detection(
#         detect_model, sr_path, score_threshold=0.45
#     )
#
#     # Scale boxes from 512 → actual SR size
#     scale_x = orig_w / 512.0
#     scale_y = orig_h / 512.0
#
#     boxes = []
#     for box in boxes_512:
#         x1, y1, x2, y2 = box
#         boxes.append([
#             x1 * scale_x,
#             y1 * scale_y,
#             x2 * scale_x,
#             y2 * scale_y
#         ])
#     boxes = np.array(boxes)
#
#     # Draw red bounding boxes (no text near them)
#     if len(boxes) > 0:
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(orig_w, x2)
#             y2 = min(orig_h, y2)
#
#             # Red box
#             color = (0, 0, 255)  # BGR red
#             thickness = max(2, int(orig_w * 0.008))  # reasonable on 1024 or 256
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
#
#     # ── Single label in top-left corner ───────────────────────────────
#     # Text format: "Fracture (xx.x%)" or "Non Fractured (xx.x%)"
#     # Assuming result comes directly from CLASS_NAMES[pred]
#     if result == "Fracture":
#         label_text = f"Fracture ({confidence:.1%})"
#         text_color = (0, 0, 255)  # red
#         bg_color = (20, 20, 60)
#     else:
#         label_text = f"Non Fractured ({confidence:.1%})"
#         text_color = (0, 255, 0)  # green
#         bg_color = (20, 60, 20)
#
#     # Font settings – adjusted for both 256×256 and 1024×1024
#     font_scale = max(0.8, orig_w / 1000.0)   # scales nicely
#     thickness  = 2 if orig_w <= 512 else 3
#
#     (text_w, text_h), baseline = cv2.getTextSize(
#         label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
#     )
#
#     # Position: top-left with small margin
#     margin = 15
#     text_x = margin
#     text_y = margin + text_h + 8   # a bit below top edge
#
#     # Background rectangle (semi-transparent look)
#     cv2.rectangle(
#         img,
#         (text_x - 10, text_y - text_h - 10),
#         (text_x + text_w + 10, text_y + 10),
#         bg_color,
#         cv2.FILLED
#     )
#
#     # White outline + colored text (better visibility)
#     cv2.putText(
#         img, label_text,
#         (text_x, text_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale, (255, 255, 255), thickness + 1  # white outline
#     )
#     cv2.putText(
#         img, label_text,
#         (text_x, text_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale, text_color, thickness
#     )
#
#     # Save result
#     cv2.imwrite(detect_path, img)
#
#     return render_template(
#         "final_result.html",
#         input_image=filename,
#         output_image=filename,
#         detect_image=filename,
#         result=result,
#         confidence=confidence
#     )
#
# @app.route("/detect_only", methods=["GET", "POST"])
# def detect_only():
#     if request.method == "POST":
#         file = request.files.get("image")
#
#         if not file or file.filename == "":
#             return "No file uploaded"
#
#         filename = secure_filename(file.filename)
#         input_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(input_path)
#
#         # Run fracture detection ONLY
#         result, confidence = run_fracture_detection(detect_model, input_path)
#
#         return render_template(
#             "detect_only.html",
#             result=result,
#             confidence=confidence
#         )
#
#     return render_template("detect_only.html")

# app.py — add this helper function near the top

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

    font_scale = max(0.8, w / 1000.0)
    thick = 2 if w <= 512 else 3
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

    x, y = 15, 15 + th + 8
    cv2.rectangle(img, (x-10, y-th-10), (x+tw+10, y+10), bg, cv2.FILLED)
    cv2.putText(img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick+1)
    cv2.putText(img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

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

    result, conf, boxes, scores = run_fracture_detection(detect_model, input_path, 0.45)
    draw_fracture_visualization(input_path, boxes, scores, result, conf, detect_path)

    return render_template("final_result.html",
                           input_image=filename,
                           output_image=None,  # no SR here
                           detect_image=filename,
                           # result=result,
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
    return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=port)

