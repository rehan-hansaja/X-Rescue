import os
from flask import Flask, render_template, request, send_from_directory
from sr_predict import run_sr
from sr_predict import load_model

UPLOAD_FOLDER = "images/uploads"
OUTPUT_FOLDER = "images/outputs"
MODEL_PATH = "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

model = load_model()  # Load once at startup

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file.filename == "":
            return "No file uploaded"

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, file.filename)

        file.save(input_path)

        run_sr(model, input_path, output_path)

        return render_template(
            "index.html",
            input_image=file.filename,
            output_image=file.filename
        )

    return render_template("index.html")

@app.route("/uploads/<filename>")
def input_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)