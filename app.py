import os
import concurrent.futures
import json
from gradio_client import Client, file
from FaceShape.faceshape import process_face_shape

# Paths
FACE_ONLY_FOLDER = "Face_Only"
OUTPUT_FOLDER = "Output"
FILE_NAME = r"FaceShape/Round.jpg"
MAP_FILE = "map.json"  # JSON file containing face shape to hairstyle mappings

# Extract the filename without extension for folder naming
face_filename = os.path.splitext(os.path.basename(FILE_NAME))[0]
USER_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, face_filename)

# Detect face shape
FACE_SHAPE = process_face_shape(FILE_NAME)
print(f"Detected Face Shape: {FACE_SHAPE}")

# Load face shape to hairstyle mapping from map.json
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, "r") as f:
        FACE_SHAPE_MAP = json.load(f)
else:
    print(f"Error: {MAP_FILE} not found!")
    exit(1)

# Check if the folder already exists
if os.path.exists(USER_OUTPUT_FOLDER):
    print(f"Folder '{USER_OUTPUT_FOLDER}' exists, skipping API calls.")
else:
    os.makedirs(USER_OUTPUT_FOLDER, exist_ok=True)  # Create folder if not exists

    client = Client("AIRI-Institute/HairFastGAN")

    result1 = client.predict(
        img=file(FILE_NAME),
        align=["Face", "Shape", "Color"],
        api_name="/resize_inner"
    )

    # Generate all filenames (35 files)
    categories = ["Buzzcut", "Crewcut", "Fringe", "Pompadour", "SidePart", "SlickedBack", "Undercut"]
    indices = [1, 2, 3]
    shape_color_files = [f"{category}{index}.webp" for category in categories for index in indices]

    def call_api(shape_color_file):
        """Function to make an API call and store the result."""
        result = client.predict(
            face=file(result1),
            shape=file(os.path.join(FACE_ONLY_FOLDER, shape_color_file)),
            color=file(os.path.join(FACE_ONLY_FOLDER, shape_color_file)),
            blending="Article",
            poisson_iters=0,
            poisson_erosion=15,
            api_name="/swap_hair"
        )

        # Save result as JSON inside Output/{face_filename}
        output_file = os.path.join(USER_OUTPUT_FOLDER, f"{os.path.splitext(shape_color_file)[0]}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

        return shape_color_file

    # Use threading to make API calls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Reduced to 10 for stability
        list(executor.map(call_api, shape_color_files))

    print(f"All API calls completed. Results saved in {USER_OUTPUT_FOLDER}.")

# --------------- FIND RELEVANT JSON FILES --------------- #

matching_keys = FACE_SHAPE_MAP[FACE_SHAPE]
print(f"Matching Hairstyle Categories: {matching_keys}")

# Get all JSON files in the output folder
json_files = [f for f in os.listdir(USER_OUTPUT_FOLDER) if f.endswith(".json")]

# Filter JSON files that start with any of the matching keys
relevant_json_files = [f for f in json_files if any(f.startswith(key) for key in matching_keys)]

# Print relevant JSON files
if relevant_json_files:
    print("\nRelevant JSON files:")
    for file in relevant_json_files:
        print(file)
else:
    print("\nNo matching JSON files found for this face shape.")




