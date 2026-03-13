import os

#TODO: update this with the actual exit region coordinates
EXIT_REGION = (674, 169, 897, 865)

#TODO: update radius around exit center for dwell detection
NEAR_EXIT_RADIUS = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_FOLDER = os.path.join(BASE_DIR, "..", "videos")
FEATURE_FOLDER = os.path.join(BASE_DIR, "..", "features")

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "sam3.pt")

