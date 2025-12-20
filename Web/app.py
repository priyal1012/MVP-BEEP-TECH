# All the needed imports 

import os
import json
import concurrent.futures
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file
from FaceShape.faceshape import process_face_shape
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/results"
FACE_ONLY_FOLDER = "Face_Only"
MAP_FILE = "map.json"


# Directory Checks
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FACE_ONLY_FOLDER, exist_ok=True)

# Client loading
client = Client("AIRI-Institute/HairFastGAN")

# Loading Face Map which tells which hairstyle is better for which face shape
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, "r") as f:
        FACE_SHAPE_MAP = json.load(f)
    print("[INFO] Face shape map loaded successfully.")
else:
    FACE_SHAPE_MAP = {}
    print("[WARNING] map.json not found! Face shape mapping is empty.")

# Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

# Upload Route
@app.route("/upload", methods=["POST"])
def upload():
    print("[INFO] Received upload request")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Check if image already processed before
    image_name = os.path.splitext(filename)[0]
    user_output_folder = os.path.join(OUTPUT_FOLDER, image_name)

    file.save(filepath)

    os.makedirs(user_output_folder, exist_ok=True)

    # Detect face shape
    face_shape = process_face_shape(filepath)
    print(filepath)
    
    # If image is processed before skip the 21 API Calls
    if os.path.exists(user_output_folder) and len(os.listdir(user_output_folder)) >= 21:
        print("ALL API CALLES SKIPPED")
        print(FACE_SHAPE_MAP[face_shape])
        return jsonify({"image_name": image_name, "face_shape": face_shape, "status": "complete"})

    try:
        # API call for moulding customer image
        result1 = client.predict(
        img=handle_file(filepath),
        align=["Face", "Shape", "Color"],
        api_name="/resize_inner"
        )
        print(result1)
    except Exception as e:
        print(f"[ERROR] Face processing API call failed: {e}")
        return jsonify({"error": "Face processing failed"}), 500

    # Hairstyle categories
    categories = ["Buzzcut", "Crewcut", "Fringe", "Pompadour", "SidePart", "SlickedBack", "Undercut"]
    indices = [1, 2, 3]
    shape_color_files = [f"{category}{index}.webp" for category in categories for index in indices]

    def call_api(shape_color_file):
        print("ALL API CALLES MADE")
        try:
            shape_path = os.path.join(FACE_ONLY_FOLDER, shape_color_file)
            color_path = os.path.join(FACE_ONLY_FOLDER, shape_color_file)

            if not os.path.exists(shape_path) or not os.path.exists(color_path):
                return

            # Swap hair API Call 

            result = client.predict(
                face = handle_file(result1),
                shape=handle_file(shape_path),
                color=handle_file(color_path),
                blending="Article",
                poisson_iters=0,
                poisson_erosion=15,
                api_name="/swap_hair"
            )

            output_file = os.path.join(user_output_folder, f"{os.path.splitext(shape_color_file)[0]}.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            print(f"[ERROR] API call failed for {shape_color_file}: {e}")

    # Run all API calls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(call_api, shape_color_files)

    return jsonify({"image_name": image_name, "face_shape": face_shape, "status": "processing"})

# Route for checking the status [API Call resolved or not]
@app.route("/status/<image_name>")
def check_status(image_name):
    """Check if all 21 hairstyle results are ready."""
    user_output_folder = os.path.join(OUTPUT_FOLDER, image_name)
    if not os.path.exists(user_output_folder):
        return jsonify({"status": "processing"})

    json_files = [f for f in os.listdir(user_output_folder) if f.endswith(".json")]
    
    if len(json_files) >= 21:
        return jsonify({"status": "complete"})
    else:
        return jsonify({"status": "processing"})

# Results route
@app.route("/results/<image_name>")
def get_results(image_name):
    """Move generated images to static folder with same name as JSON file and return their URLs."""
    user_output_folder = os.path.join(OUTPUT_FOLDER, image_name)

    if not os.path.exists(user_output_folder):
        return jsonify({"error": "Results not found"}), 404

    json_files = [f for f in os.listdir(user_output_folder) if f.endswith(".json")]
    image_urls = []
    suggested_hairstyles = []

    static_folder = os.path.join("static", "results", image_name)
    os.makedirs(static_folder, exist_ok=True)

    uploaded_files = os.listdir(UPLOAD_FOLDER)
    file_path = None

    for file in uploaded_files:
        if file.startswith(image_name):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            break

    if not file_path:
        return jsonify({"error": "Uploaded file not found"}), 400

    # Finding the best match hairstyles
    FACE_SHAPE = process_face_shape(file_path)
    matching_keys = FACE_SHAPE_MAP[FACE_SHAPE]

    print("Detected Face Shape:", FACE_SHAPE)
    print(matching_keys)

    for json_file in json_files:
        json_path = os.path.join(user_output_folder, json_file)
        
        with open(json_path, "r") as f:
            data = json.load(f)
            for entry in data:
                if entry["visible"] and "value" in entry:
                    tmp_path = entry["value"]  # Path from /tmp/gradio
                    if os.path.exists(tmp_path):
                        new_image_name = os.path.splitext(json_file)[0] + ".webp"  # Match JSON name
                        new_path = os.path.join(static_folder, new_image_name)
                        shutil.copy(tmp_path, new_path)
                        image_urls.append(f"/{new_path}")

                        # The best hairstyle according to the face are added to suggested list
                        for key in matching_keys:
                            if new_image_name.startswith(key):
                                suggested_hairstyles.append(os.path.splitext(new_image_name)[0])

    return jsonify({"images": image_urls, "suggested": suggested_hairstyles, "status": "complete"})


if __name__ == "__main__":
    app.run(debug=False)