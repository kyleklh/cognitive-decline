#find right coordinates for exit region and test by drawing on video frames

import cv2
import os
import config
import utils

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"x={x}, y={y}")

# pick the first video
videos = os.listdir(config.VIDEO_FOLDER)

video_path = None
for v in videos:
    if v.lower().endswith((".mp4", ".mov")):
        video_path = os.path.join(config.VIDEO_FOLDER, v)
        break

if video_path is None:
    print("No video found")
    exit()

print(f"Testing exit region on: {video_path}")

cap = cv2.VideoCapture(video_path)

# exit region
x1, y1, x2, y2 = config.EXIT_REGION

# exit center
exit_cx, exit_cy = utils.exit_center(config.EXIT_REGION)

# create window and attach mouse tracker
cv2.namedWindow("Exit Region Debug")
cv2.setMouseCallback("Exit Region Debug", mouse_callback)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # draw exit region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

    # draw exit center
    cv2.circle(frame, (int(exit_cx), int(exit_cy)), 8, (0,0,255), -1)

    # label
    cv2.putText(frame, "EXIT REGION", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Exit Region Debug", frame)

    # press q to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()