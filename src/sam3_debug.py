import os
import cv2
import numpy as np
from ultralytics.models.sam import SAM3VideoSemanticPredictor

import config
import utils

#init SAM3
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=640,
    model=config.MODEL_PATH,
    half=True,
    save=False,
    device=0
)

predictor = SAM3VideoSemanticPredictor(overrides=overrides)

video_path = os.path.join(config.VIDEO_FOLDER, "IMG_8304.MOV")

results = predictor(source=video_path, text=["person"], stream=True)

for r in results:

    frame = r.orig_img.copy()

    #draw exit region
    x1, y1, x2, y2 = config.EXIT_REGION
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    if r.masks is not None and len(r.masks.data) > 0:

        mask = r.masks.data[0].cpu().numpy()

        mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
        mask_binary = (mask > 0.5).astype(np.uint8)

        #overlay mask
        overlay = frame.copy()
        overlay[mask_binary == 1] = (0,0,255)

        frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

        centroid = utils.mask_centroid(mask_binary)

        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 6, (255,0,0), -1)

    cv2.imshow("Mask Debug", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()