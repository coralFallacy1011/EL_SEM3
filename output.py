from ultralytics import YOLO
import cv2
import threading
import numpy as np

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

    def draw(self, results, detected_lights):
        # Draw bounding boxes for vehicles and detected lights (headlights/taillights)
        for box in results:
            x1, y1, x2, y2, score, class_id = box
            label = self.vehicle_classes.get(int(class_id), 'Unknown')
            color = (0, 255, 0)
            # Uncomment the next two lines to disable bounding box drawing
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(self.frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw bounding boxes around detected headlights or taillights
        for (x1, y1, x2, y2) in detected_lights:
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for lights

    def manipulate_grid(self, grid_coords):
        # Create ranges for grid manipulation
        manipulated_data = []
        for (grid_x, grid_y) in grid_coords:
            # Example logic: Turn off 1 row above and below and 1 column left and right
            w = max(0, grid_y - 1)  # Row start
            x = min(self.grid_rows - 1, grid_y + 1)  # Row end
            y = max(0, grid_x - 1)  # Column start
            z = min(self.grid_cols - 1, grid_x + 1)  # Column end
            manipulated_data.append((w, x, y, z))

        return manipulated_data

    def display(self):
        while self.cap.isOpened() and self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Detect vehicles via YOLO and map to grid
            vehicles, vehicle_grid_coords = self.process(self.frame)

            # Detect headlights/taillights based on brightness
            detected_lights = self.detect_headlights_taillights(self.frame)

            # Manipulate grid based on detected coordinates
            manipulated_data = self.manipulate_grid(vehicle_grid_coords)

            # Print manipulated data for Arduino
            print(f"Manipulated Data (to send to Arduino): {manipulated_data}")

            # Draw the results on the frame
            self.draw(vehicles, detected_lights)

            # Display the frame with detections
            cv2.imshow("Vehicle Detection and Grid Manipulation", self.frame)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def terminate(self):
        # Release resources and close windows
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("Vehicle Detection terminated.")

# Helper function to start the vehicle detection in a thread
def start_detection(detector):
    detector.display()

if __name__ == '__main__':
    # Set the video path
    video_path = "/home/coralfallacy/Desktop/SEM3EL/dashcam.mp4"

    # Initialize the VehicleDetection instance with the video file
    VideoDetection = VehicleDetectionAndGridManipulation(video_path)

    # Create and start a thread for video detection
    thread = threading.Thread(target=start_detection, args=(VideoDetection,))
    thread.start()

    # Wait for the thread to finish
    thread.join()

    # Terminate the detector after thread completes
    VideoDetection.terminate()
