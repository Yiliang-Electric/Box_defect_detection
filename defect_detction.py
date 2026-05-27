import torch
import numpy as np
import cv2
from ultralytics import YOLO
import pyzed.sl as sl

class ObjectDetection:
    def __init__(self):
        # Choose device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load your YOLO model (replace "box_defect.pt" with your weights)
        self.model = YOLO("box_defect.pt")
        self.model.fuse()

        # Set up ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Failed to open ZED camera. Error code: {err}")
        self.image = sl.Mat()  # container for each frame

    def __call__(self):
        try:
            while True:
                # Grab a frame from ZED
                if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image (RGBA) and convert to RGB
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    raw_rgba = self.image.get_data()
                    rgb_frame = cv2.cvtColor(raw_rgba, cv2.COLOR_RGBA2RGB)

                    # Run YOLO inference on the RGB frame
                    results = self.model(rgb_frame)

                    # Convert to BGR for OpenCV drawing
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                    # Draw only bounding boxes and centers
                    annotated = self.draw_bboxes_and_centers(bgr_frame, results)

                    # Display result
                    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
                    cv2.imshow("Detection", annotated)

                    # Exit on ESC
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            self.zed.close()
            cv2.destroyAllWindows()

    def draw_bboxes_and_centers(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Given a BGR frame and YOLO results, draw each bounding box
        and a small circle at its center. No labels or confidences.
        """
        boxes = results[0].boxes  # single‐image batch
        xyxy = boxes.xyxy.cpu().numpy().astype(int)  # shape: (N, 4)

        for (x1, y1, x2, y2) in xyxy:
            # Draw rectangle (red, 2 px thick)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Compute center of the box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw a small filled circle (green) at the center
            cv2.circle(frame, (cx, cy), radius=4, color=(0, 255, 0), thickness=-1)

        return frame

if __name__ == "__main__":
    detector = ObjectDetection()
    detector()
