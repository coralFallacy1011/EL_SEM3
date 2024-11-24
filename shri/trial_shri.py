import cv2
import numpy as np
import time
import math
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
cap = cv2.VideoCapture("S:\\SmartBeamX\\nightdriving.mp4")

KNOWN_WIDTH = 1.8  
FOCAL_LENGTH = 700  
vehicle_positions = {}
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [2, 3, 5, 7]: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers.append((center_x, center_y))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    curr_time = time.time()
    delta_time = curr_time - prev_time

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w  

            vehicle_id = f"{label}-{i}"
            if vehicle_id in vehicle_positions:
                prev_center, prev_distance = vehicle_positions[vehicle_id]
                pixel_displacement = math.sqrt((centers[i][0] - prev_center[0])**2 + (centers[i][1] - prev_center[1])**2)
                speed = (pixel_displacement / delta_time) * (distance / FOCAL_LENGTH)
            else:
                speed = 0  
            vehicle_positions[vehicle_id] = (centers[i], distance)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

    cv2.imshow("Vehicle Detection", frame)
    prev_time = curr_time

    if cv2.waitKey(1) == 27: 
        break
cap.release()
cv2.destroyAllWindows()
