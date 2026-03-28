import os
import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory, flash
from werkzeug.utils import secure_filename
from sr_predict import run_sr, load_model as load_sr_model
from detect_predict import run_fracture_detection, load_detection_model

# Load environment variables from .env file (for EmailJS credentials)
load_dotenv()

# FILE VALIDATION CONFIGURATION
# Define allowed image extensions for upload validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif', 'webp', 'bmp', 'tiff'}

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed image extension.
    Args:
        filename (str): Name of the uploaded file
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# FOLDER CONFIGURATION
# Define directories for storing uploaded and processed images
UPLOAD_FOLDER = "images/uploads"  # Original uploaded images
OUTPUT_FOLDER = "images/outputs"  # Super-resolution enhanced images
DETECT_FOLDER = "images/detections"  # Fracture detection visualizations

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

# Initialize Flask application
app = Flask(__name__)
# Set a secret key for flash messages (required for flashing)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# LOAD ML MODELS
# Load super-resolution model once at startup
sr_model = load_sr_model()
# Load fracture detection model once at startup
detect_model = load_detection_model(os.path.join(os.path.dirname(__file__), "models", "best_fracture_detection.pth"))


# ROUTE: HOME PAGE
@app.route("/")
def home():
    """Render the home/landing page with example images."""
    return render_template("index.html")

# ROUTE: SUPER-RESOLUTION + DETECTION WORKFLOW
@app.route("/sr_detect", methods=["GET", "POST"])
def sr_detect():
    """
    Handle super-resolution enhancement followed by fracture detection.
    GET: Display upload form
    POST: Process uploaded image, run super-resolution, and show result
    """
    if request.method == "POST":
        # Get uploaded file from form
        file = request.files.get("image")

        # Check if file was uploaded
        if not file or file.filename == "":
            flash("No file selected. Please upload an image.", "error")
            return render_template("sr_detect.html")

        # Validate file format
        if not allowed_file(file.filename):
            flash("Invalid file format! Please upload an image file (PNG, JPG, JPEG, JFIF, WEBP, BMP, TIFF).", "error")
            return render_template("sr_detect.html")

        # Secure the filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save uploaded file
        file.save(input_path)

        # Verify the file is actually a valid image
        try:
            img_temp = cv2.imread(input_path)
            if img_temp is None:
                flash("Uploaded file is corrupted or not a valid image.", "error")
                os.remove(input_path)  # Clean up invalid file
                return render_template("sr_detect.html")
        except Exception as e:
            flash(f"Error reading image: {str(e)}", "error")
            if os.path.exists(input_path):
                os.remove(input_path)
            return render_template("sr_detect.html")

        # Handle JFIF format
        if filename.lower().endswith('.jfif'):
            new_filename = filename.rsplit('.', 1)[0] + '.jpg'
            new_input_path = os.path.join(UPLOAD_FOLDER, new_filename)
            cv2.imwrite(new_input_path, img_temp)
            os.remove(input_path)
            input_path = new_input_path
            filename = new_filename

        # Define output path for super-resolution result
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Run super-resolution enhancement
        run_sr(sr_model, input_path, output_path)

        # Render template with input and output images
        return render_template(
            "sr_detect.html",
            input_image=filename,
            output_image=filename
        )

    # GET request: Show upload form
    return render_template("sr_detect.html")

# VISUALIZATION HELPER FUNCTION
def draw_fracture_visualization(img_path, boxes_512, scores, result, confidence, output_path):
    """
    Draw bounding boxes and labels on the image for fracture detection results.
    Args:
        img_path (str): Path to input image
        boxes_512 (np.ndarray): Bounding boxes in 512x512 coordinate space
        scores (np.ndarray): Confidence scores for each box
        result (str): "Fracture" or "No Fracture"
        confidence (float): Overall confidence score
        output_path (str): Path to save visualized image
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h, w = img.shape[:2]

    # Scale boxes from 512x512 space to original image dimensions
    scale_x = w / 512.0
    scale_y = h / 512.0
    boxes = np.array([
        [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        for x1, y1, x2, y2 in boxes_512
    ])

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    # Keeps only the highest confidence boxes that overlap less than threshold
    if len(boxes) > 0:
        import torch
        import torchvision
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
        boxes = boxes_tensor[keep].numpy()
        scores = scores_tensor[keep].numpy()

    # Draw bounding boxes on image
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Clip to image boundaries
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            # Draw red rectangle (BGR color: 0,0,255)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, int(w * 0.008)))

    # DRAW STATUS LABEL
    # Determine label text and colors based on detection result
    if result == "Fracture":
        label = f"Fracture ({confidence:.1%})"
        color = (0, 0, 255)  # Red text for fracture
        bg = (20, 20, 60)  # Dark red background
    else:
        label = f"Non Fractured ({confidence:.1%})"
        color = (0, 255, 0)  # Green text for no fracture
        bg = (20, 60, 20)  # Dark green background

    # Scale font size based on image width
    font_scale = max(0.4, w / 800.0)
    thickness = 2 if w <= 300 else max(1, int(w / 300))

    # Get text dimensions for background rectangle
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Position label in top-left corner with padding
    x, y = 15, 15 + th + 8

    # Draw background rectangle
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), bg, cv2.FILLED)

    # Draw text outline for better visibility on large images
    if w > 512:
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1)
    # Draw main text
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # Save visualized image
    cv2.imwrite(output_path, img)


# ROUTE: CONTINUE TO FRACTURE DETECTION
@app.route("/detect_fracture", methods=["POST"])
def detect_fracture():
    """
    Handle the second step of SR+Detection workflow.
    Takes super-resolution output and runs fracture detection.
    """
    # Get filename from hidden form input
    filename = request.form["filename"]
    sr_path = os.path.join(OUTPUT_FOLDER, filename)
    detect_path = os.path.join(DETECT_FOLDER, filename)

    # Run fracture detection on enhanced image
    result, conf, boxes, scores = run_fracture_detection(detect_model, sr_path, 0.6)

    # Draw visualization with bounding boxes and labels
    draw_fracture_visualization(sr_path, boxes, scores, result, conf, detect_path)

    # Render final result page
    return render_template("final_result.html",
                           input_image=filename,
                           output_image=filename,  # SR output available
                           detect_image=filename,
                           confidence=conf
                           )


# ROUTE: FRACTURE DETECTION ONLY
@app.route("/detect_only", methods=["GET", "POST"])
def detect_only():
    """
    Handle standalone fracture detection.
    GET: Display upload form
    POST: Process uploaded image and show detection results
    """
    if request.method == "GET":
        return render_template("detect_only.html")

    file = request.files.get("image")

    # Check if file was uploaded
    if not file or not file.filename:
        flash("No file selected. Please upload an image.", "error")
        return render_template("detect_only.html")

    # Validate file format
    if not allowed_file(file.filename):
        flash("Invalid file format! Please upload an image file (PNG, JPG, JPEG, JFIF, WEBP, BMP, TIFF).", "error")
        return render_template("detect_only.html")

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    detect_path = os.path.join(DETECT_FOLDER, filename)

    # Save uploaded file
    file.save(input_path)

    # Verify the file is actually a valid image
    try:
        img_temp = cv2.imread(input_path)
        if img_temp is None:
            flash("Uploaded file is corrupted or not a valid image.", "error")
            if os.path.exists(input_path):
                os.remove(input_path)
            return render_template("detect_only.html")
    except Exception as e:
        flash(f"Error reading image: {str(e)}", "error")
        if os.path.exists(input_path):
            os.remove(input_path)
        return render_template("detect_only.html")

    # Handle JFIF format conversion
    if filename.lower().endswith('.jfif'):
        new_filename = filename.rsplit('.', 1)[0] + '.jpg'
        new_input_path = os.path.join(UPLOAD_FOLDER, new_filename)
        cv2.imwrite(new_input_path, img_temp)
        os.remove(input_path)
        input_path = new_input_path
        filename = new_filename
        detect_path = os.path.join(DETECT_FOLDER, filename)

    # Run fracture detection on original image
    result, conf, boxes, scores = run_fracture_detection(detect_model, input_path, 0.6)

    # Draw visualization
    draw_fracture_visualization(input_path, boxes, scores, result, conf, detect_path)

    # Render final result page (no SR output)
    return render_template("final_result.html",
                           input_image=filename,
                           output_image=None,  # No SR output for this workflow
                           detect_image=filename,
                           confidence=conf
                           )


# STATIC FILE SERVING ROUTES
@app.route("/uploads/<filename>")
def input_file(filename):
    """Serve uploaded input images."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    """Serve super-resolution output images."""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/detections/<filename>")
def detect_file(filename):
    """Serve fracture detection visualization images."""
    return send_from_directory(DETECT_FOLDER, filename)

# ROUTE: CONTACT PAGE
@app.route("/contact")
def contact():
    """
    Render contact page with EmailJS credentials.
    Credentials are injected from environment variables for security.
    """
    key = os.environ.get("EMAILJS_PUBLIC_KEY")
    service = os.environ.get("EMAILJS_SERVICE_ID")
    template = os.environ.get("EMAILJS_TEMPLATE_ID")

    return render_template("contact.html",
                           emailjs_public_key=key,
                           emailjs_service_id=service,
                           emailjs_template_id=template
                           )

# APPLICATION ENTRY POINT
if __name__ == "__main__":
    # Get port from environment variable
    port = int(os.environ.get("PORT", 7860))

    # Run app on all interfaces (0.0.0.0) to make it accessible externally
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=port)

