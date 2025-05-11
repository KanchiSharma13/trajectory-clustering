# Importing Modules
import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Loading Models
model = YOLO("yolov8n.pt")

# Loading Video 
cap = cv2.VideoCapture("traffic.mp4")
# cap = cv2.VideoCapture(0)

track_history = defaultdict(lambda: [])
# Read Frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Make detections 
        results = model.track(frame,persist=True,tracker="bytetrack.yaml", conf=0.5)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow('YOLO', annotated_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print(track_history)

import pickle
pickle.dump(track_history,open("track.pkl",'wb'))
# frame = -1
# ret = True
# while(ret):
#     ret, frame = cap.read()
#     if(ret and frame < 10):
#         results = model(frame)[0]
#         for result in results.boxes.data.tolist:
#             print(result)


