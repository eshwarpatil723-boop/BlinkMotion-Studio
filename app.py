
import os
import uuid

import cv2
import numpy as np
import mediapipe as mp

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename


# ==========================================================
# APPLICATION CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "blinkmotion-local-development-key"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ==========================================================
# VIDEO SETTINGS
# ==========================================================

FPS = 24
VIDEO_DURATION = 4
TOTAL_FRAMES = FPS * VIDEO_DURATION

# Five intended blink events during the video.
BLINK_CENTERS = [0.40, 1.15, 1.90, 2.65, 3.40]

# MediaPipe Face Mesh eye landmark indices.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


# ==========================================================
# IMAGE VALIDATION
# ==========================================================

def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


# ==========================================================
# LANDMARK COORDINATES
# ==========================================================

def get_point(landmarks, index, width, height):
    """Convert normalized facial landmarks into image coordinates."""

    landmark = landmarks[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


# ==========================================================
# BLINK TIMING
# ==========================================================

def blink_intensity(time_value, blink_centers):
    """
    Calculate blink intensity.

    0.0 means no effect.
    1.0 means maximum attempted eye closure.
    """

    blink_duration = 0.22

    for center in blink_centers:
        start = center - blink_duration / 2
        end = center + blink_duration / 2

        if start <= time_value <= end:
            progress = (time_value - start) / blink_duration

            if progress < 0.5:
                intensity = progress / 0.5
            else:
                intensity = (1.0 - progress) / 0.5

            intensity = max(0.0, min(1.0, intensity))

            # Smooth the closing and opening motion.
            return intensity * intensity * (3 - 2 * intensity)

    return 0.0


# ==========================================================
# APPROXIMATE EYE-CLOSING EFFECT
# ==========================================================

def draw_closed_eye(frame, landmarks, indices, intensity):
    """
    Approximate eye closure using facial landmarks and OpenCV.

    Note:
    This is an image-processing effect, not a generative AI model.
    Results depend on the portrait and may not look fully realistic.
    """

    if intensity <= 0.02:
        return

    height, width = frame.shape[:2]

    points = [
        get_point(landmarks, index, width, height)
        for index in indices
    ]

    corner_left = np.array(points[0], dtype=np.float32)
    corner_right = np.array(points[3], dtype=np.float32)

    eye_width = float(np.linalg.norm(corner_right - corner_left))

    if eye_width < 5:
        return

    center = (corner_left + corner_right) / 2.0

    center_x = int(center[0])
    center_y = int(center[1])

    # Create a mask around the eye.
    mask = np.zeros((height, width), dtype=np.uint8)

    radius_x = max(4, int(eye_width * 0.62))
    radius_y = max(3, int(eye_width * 0.23))

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        360,
        255,
        -1,
    )

    # Expand the mask slightly.
    kernel_size = max(3, int(eye_width * 0.06))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    mask = cv2.dilate(mask, kernel, iterations=1)

    # Reconstruct the masked region from surrounding pixels.
    repaired = cv2.inpaint(
        frame,
        mask,
        3,
        cv2.INPAINT_TELEA,
    )

    # Blend gradually during the blink.
    alpha = max(0.0, min(1.0, float(intensity)))

    blended = cv2.addWeighted(
        repaired,
        alpha,
        frame,
        1.0 - alpha,
        0,
    )

    frame[:] = blended

    # Draw a subtle curved eyelid line.
    left_x, left_y = map(int, corner_left)
    right_x, right_y = map(int, corner_right)

    middle_x = (left_x + right_x) // 2
    middle_y = (left_y + right_y) // 2

    curve_depth = max(1, int(eye_width * 0.08 * intensity))

    curve = np.array(
        [
            [left_x, left_y],
            [middle_x, middle_y + curve_depth],
            [right_x, right_y],
        ],
        dtype=np.int32,
    )

    sample_x = max(0, min(width - 1, center_x))
    sample_y = max(0, min(height - 1, center_y))

    pixel = frame[sample_y, sample_x]

    line_color = tuple(
        int(channel * 0.75)
        for channel in pixel
    )

    thickness = max(1, int(width / 500))

    cv2.polylines(
        frame,
        [curve],
        False,
        line_color,
        thickness,
        cv2.LINE_AA,
    )


# ==========================================================
# GENERATE BLINK VIDEO
# ==========================================================

def generate_blink_video(image_path, output_path):
    """Detect facial landmarks and generate a four-second MP4."""

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "The image could not be opened. "
            "Please upload a valid JPG, PNG, or WEBP image."
        )

    height, width = image.shape[:2]

    # Resize very large images.
    max_dimension = 1280

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        height, width = image.shape[:2]

    # Convert BGR to RGB for MediaPipe.
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # Detect the face.
    face_mesh_module = mp.solutions.face_mesh

    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError(
            "No face was detected. Upload a clear portrait "
            "with the face looking toward the camera."
        )

    landmarks = results.multi_face_landmarks[0].landmark

    # Initialize the MP4 video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        FPS,
        (width, height),
    )

    if not video_writer.isOpened():
        video_writer.release()

        raise RuntimeError(
            "OpenCV could not create the MP4 file."
        )

    try:
        for frame_number in range(TOTAL_FRAMES):
            current_time = frame_number / FPS

            # Start with the original portrait for every frame.
            frame = image.copy()

            intensity = blink_intensity(
                current_time,
                BLINK_CENTERS,
            )

            # Apply the approximate effect to both eyes.
            draw_closed_eye(
                frame,
                landmarks,
                LEFT_EYE_INDICES,
                intensity,
            )

            draw_closed_eye(
                frame,
                landmarks,
                RIGHT_EYE_INDICES,
                intensity,
            )

            video_writer.write(frame)

    finally:
        video_writer.release()

    if (
        not os.path.isfile(output_path)
        or os.path.getsize(output_path) == 0
    ):
        raise RuntimeError(
            "Video generation failed. No valid MP4 was created."
        )

    return output_path


# ==========================================================
# HOME PAGE AND UPLOAD
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template(
            "index.html",
            generated=False,
        )

    if "image" not in request.files:
        flash("Please select an image.")
        return redirect(url_for("index"))

    uploaded_file = request.files["image"]

    if not uploaded_file or not uploaded_file.filename:
        flash("Please choose an image first.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Upload a JPG, JPEG, PNG, or WEBP image.")
        return redirect(url_for("index"))

    safe_name = secure_filename(uploaded_file.filename)

    extension = os.path.splitext(safe_name)[1].lower()

    unique_id = uuid.uuid4().hex

    input_filename = f"{unique_id}{extension}"
    output_filename = f"blinkmotion_{unique_id}.mp4"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename,
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename,
    )

    try:
        uploaded_file.save(input_path)

        generate_blink_video(
            input_path,
            output_path,
        )

        return render_template(
            "index.html",
            generated=True,
            video_url=url_for(
                "download_file",
                filename=output_filename,
            ),
        )

    except Exception as error:
        app.logger.exception(
            "BlinkMotion video generation failed"
        )

        flash(f"Could not generate the video: {error}")

        return redirect(url_for("index"))

    finally:
        # Delete the temporary uploaded image.
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                app.logger.warning(
                    "Could not remove the temporary image."
                )


# ==========================================================
# DOWNLOAD VIDEO
# ==========================================================

@app.route("/download/<path:filename>")
def download_file(filename):

    if not filename.lower().endswith(".mp4"):
        return "Invalid file type.", 400

    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        as_attachment=True,
    )


# ==========================================================
# HANDLE LARGE UPLOADS
# ==========================================================

@app.errorhandler(413)
def file_too_large(error):
    flash(
        "The image is too large. Please upload an image "
        "smaller than 15 MB."
    )

    return redirect(url_for("index"))


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("BLINKMOTION STUDIO")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )