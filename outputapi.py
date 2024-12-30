from ultralytics import YOLO
import cv2
import threading
import numpy as np
from flask import Flask, jsonify, request

# Flask app
app = Flask(__name__)

class VehicleDetectionAndGridManipulation:
    def __init__(self, video_source):
        # Load YOLO model for vehicle detection
        self.model = YOLO("yolov8n.pt")
        self.vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}
        self.cap = cv2.VideoCapture(video_source)
        self.running = True

        # Define grid dimensions
        self.grid_rows = 8
        self.grid_cols = 16

        # Lock for thread synchronization
        self.lock = threading.Lock()

        # Shared data for the Flask API
        self.manipulated_data = []

    def process(self, frame):
        # Perform inference and filter for vehicle classes
        results = self.model(frame)
        filtered_results = [r for r in results[0].boxes.data if int(r[-1]) in self.vehicle_classes]

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Initialize grid mapping results
        vehicle_grid_coords = []
        for box in filtered_results:
            x1, y1, x2, y2, score, class_id = box

            # Compute the center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Map center to grid cell
            grid_x = int((center_x / frame_width) * self.grid_cols)
            grid_y = int((center_y / frame_height) * self.grid_rows)

            # Ensure grid indices are within bounds
            grid_x = max(0, min(self.grid_cols - 1, grid_x))
            grid_y = max(0, min(self.grid_rows - 1, grid_y))

            vehicle_grid_coords.append((grid_x, grid_y))

        return filtered_results, vehicle_grid_coords

    def detect_headlights_taillights(self, frame):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (15, 15), 0)

        # Thresholding to detect bright areas (headlights or taillights)
        _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours of bright spots that could be headlights or taillights
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_lights = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours (noise)
                x, y, w, h = cv2.boundingRect(contour)
                detected_lights.append((x, y, x + w, y + h))  # Store the bounding box

        return detected_lights

    def manipulate_grid(self, grid_coords):
        # Create ranges for grid manipulation
        manipulated_data = []
        for (grid_x, grid_y) in grid_coords:
            w = max(0, grid_y - 1)  # Row start
            x = min(self.grid_rows - 1, grid_y + 1)  # Row end
            y = max(0, grid_x - 1)  # Column start
            z = min(self.grid_cols - 1, grid_x + 1)  # Column end
            manipulated_data.append((w, x, y, z))

        return manipulated_data

    def display(self):
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            with self.lock:
                # Detect vehicles via YOLO and map to grid
                vehicles, vehicle_grid_coords = self.process(frame)

                # Detect headlights/taillights based on brightness
                detected_lights = self.detect_headlights_taillights(frame)

                # Manipulate grid based on detected coordinates
                self.manipulated_data = self.manipulate_grid(vehicle_grid_coords)

                # Draw detections on the frame
                for box in vehicles:
                    x1, y1, x2, y2, score, class_id = box
                    label = self.vehicle_classes.get(int(class_id), 'Unknown')
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw bounding boxes around detected headlights or taillights
                for (x1, y1, x2, y2) in detected_lights:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Display the frame locally
                cv2.imshow("Vehicle Detection and Lights", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.terminate()

    def terminate(self):
        # Release resources
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("Vehicle Detection terminated.")

# Initialize the VehicleDetection instance with a video file
video_path = "/home/vipul-s/Desktop/sem3el/dashcamvid.mp4"
vehicle_detector = VehicleDetectionAndGridManipulation(video_path)

# Background thread to continuously process video
def detection_thread():
    vehicle_detector.display()

detection_thread = threading.Thread(target=detection_thread)
detection_thread.start()

# Flask API endpoint to get manipulated grid data
@app.route('/api/grid-data', methods=['GET'])
def get_grid_data():
    with vehicle_detector.lock:
        return jsonify({'grid_data': vehicle_detector.manipulated_data})

# Terminate the detection and release resources on shutdown
@app.route('/api/terminate', methods=['POST'])
def terminate():
    vehicle_detector.terminate()
    return jsonify({'message': 'Vehicle Detection terminated.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

