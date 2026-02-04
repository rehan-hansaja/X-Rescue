import os
from flask import Flask, render_template, request, send_from_directory
from sr_predict import run_sr
from sr_predict import load_model
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "images/uploads"
OUTPUT_FOLDER = "images/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

model = load_model()  # Load once at startup


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
        run_sr(model, input_path, output_path)

        return render_template(
            "sr_detect.html",
            input_image=filename,
            output_image=filename
        )

    return render_template("sr_detect.html")


@app.route("/detect_only", methods=["GET", "POST"])
def detect_only():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return "No file uploaded"

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Run fracture detection ONLY
        result, confidence = run_fracture_detection(model_detect, input_path)

        return render_template(
            "detect_only.html",
            result=result,
            confidence=confidence
        )

    return render_template("detect_only.html")


@app.route("/uploads/<filename>")
def input_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=port)
