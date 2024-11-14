from ultralytics import YOLO
import cv2
import threading

class VehicleDetection:
    def __init__(self, video_source):
        # Load YOLO model for vehicle detection
        self.model = YOLO("yolov8n.pt")
        self.vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}
        self.cap = cv2.VideoCapture(video_source)
        self.running = True

    def process(self, frame):
        # Perform inference and filter for vehicle classes
        results = self.model(frame)
        filtered_results = [r for r in results[0].boxes.data if int(r[-1]) in self.vehicle_classes]
        return filtered_results

    def draw(self, results):
        # Draw bounding boxes and labels on detected vehicles
        for box in results:
            x1, y1, x2, y2, score, class_id = box
            label = self.vehicle_classes.get(int(class_id), 'Unknown')
            color = (0, 255, 0)
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(self.frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def display(self):
        while self.cap.isOpened() and self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Convert to grayscale and enhance for better night-time visibility
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            enhanced_frame = cv2.equalizeHist(gray_frame)
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

            # Detect vehicles and draw boxes
            vehicles = self.process(enhanced_frame)
            self.draw(vehicles)

            # Display the frame with detections
            cv2.imshow("Vehicle Detection", self.frame)

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
    video_path = "/home/coralfallacy/Desktop/SEM3EL/trialvid.mp4"

    # Initialize the VehicleDetection instance with the video file
    VideoDetection = VehicleDetection(video_path)

    # Create and start a thread for video detection
    thread = threading.Thread(target=start_detection, args=(VideoDetection,))
    thread.start()

    # Wait for the thread to finish
    thread.join()

    # Terminate the detector after thread completes
    VideoDetection.terminate()
