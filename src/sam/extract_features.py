import csv
import os
import sys

import cv2
from ultralytics.models.sam import SAM3VideoSemanticPredictor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
import utils

# Create output folder if it does not exist.
os.makedirs(config.FEATURE_FOLDER, exist_ok=True)

# Compute exit center.
exit_cx, exit_cy = utils.exit_center(config.EXIT_REGION)

# Init SAM3.
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=640,
    model=config.MODEL_PATH,
    half=True,
    save=False,
    device=0,
)

predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Loop through videos.
for video in os.listdir(config.VIDEO_FOLDER):
    if not video.lower().endswith((".mp4", ".mov")):
        continue

    video_path = os.path.join(config.VIDEO_FOLDER, video)
    csv_path = os.path.join(config.FEATURE_FOLDER, os.path.splitext(video)[0] + ".csv")

    # Skip already processed videos.
    if os.path.exists(csv_path):
        print("Skipping already processed video:", video)
        continue

    print(f"Extracting from {video}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    dt = (1.0 / fps if fps > 0 else 1 / 30) * 5  # Multiply by 5 for frame skip.

    prev_d = None
    prev_cx = None
    prev_cy = None
    dwell_time = 0
    entry_count = 0
    was_near = False

    results = predictor(source=video_path, text=["person"], stream=True)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_t", "tau_t", "n_t", "v_t"])

        frame_count = 0
        for r in results:
            frame_count += 1

            # Sample every 5 frames to speed up.
            if frame_count % 5 != 0:
                continue

            d_t = None
            v_t = 0

            if r.masks is not None and len(r.masks.data) > 0:
                mask = r.masks.data[0].cpu().numpy()

                centroid = utils.mask_centroid(mask)
                if centroid is not None:
                    cx, cy = centroid

                    # Moving average smoothing.
                    if prev_cx is not None:
                        cx = 0.7 * prev_cx + 0.3 * cx
                        cy = 0.7 * prev_cy + 0.3 * cy

                    prev_cx = cx
                    prev_cy = cy

                    # Distance to exit.
                    d_t = utils.distance(cx, cy, exit_cx, exit_cy)

                    near = d_t <= config.NEAR_EXIT_RADIUS
                    if near:
                        dwell_time += dt
                    else:
                        dwell_time = 0

                    if (not was_near) and near:
                        entry_count += 1

                    was_near = near

                    if prev_d is not None:
                        v_t = (prev_d - d_t) / dt

                    prev_d = d_t

            writer.writerow([d_t, dwell_time, entry_count, v_t])

print("Feature extraction complete")