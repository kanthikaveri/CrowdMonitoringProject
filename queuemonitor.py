from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------------
# Load YOLO Model
# -------------------------------
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded.")

# ------------------------------------
# Queue Selection Variables (Global)
# ------------------------------------
queue_points = []


def mouse_callback(event, x, y, flags, param):

    global queue_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(queue_points) < 8:

            queue_points.append((x, y))
            print("Selected:", queue_points)


# ==================================================
# Queue Processing Function
# ==================================================
def process_queue(input_path, output_path):

    # IMPORTANT
    global queue_points

    print("Entered process_queue()")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error opening video")
        return 0, "0m 0s"

    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        20,
        (width, height)
    )

    print("VideoWriter created:", out.isOpened())

    if not out.isOpened():
        print("Error: Could not create output video.")
        cap.release()
        return 0, "0m 0s"

    max_queue = 0
    frame_no = 0

    cv2.namedWindow("AI Queue Monitoring")
    cv2.setMouseCallback("AI Queue Monitoring", mouse_callback)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Finished reading video.")
            break

        frame_no += 1

        frame = cv2.resize(frame, (width, height))

        polygon_ready = False

        # Draw selected points
        for point in queue_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Draw polygon after selecting at least 3 points
        if len(queue_points) >= 3:

            polygon_ready = True

            queue_polygon = np.array(queue_points, np.int32)

            cv2.polylines(
                frame,
                [queue_polygon],
                True,
                (255, 0, 0),
                3
            )

        if frame_no % 30 == 0:
            print(f"Processing Frame {frame_no}")

        # Remaining code continues below...
        try:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.20,
                verbose=False
            )
        except Exception as e:
            print("Tracking Error:", e)
            break

        queue_count = 0

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside = -1
                if polygon_ready:
                    inside = cv2.pointPolygonTest(
                        queue_polygon,
                        (center_x, center_y),
                        False
                    )

                if inside >= 0:
                    color = (0, 255, 0)
                    queue_count += 1
                    if queue_count > max_queue:
                        max_queue = queue_count
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)

                if box.id is not None:
                    track_id = int(box.id.item())
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        waiting_seconds = queue_count * 20
        minutes = waiting_seconds // 60
        seconds = waiting_seconds % 60

        cv2.rectangle(frame, (10, 10), (350, 130), (40, 40, 40), -1)

        cv2.putText(
            frame,
            f"Queue Length : {queue_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Waiting Time : {minutes}m {seconds}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Click around queue (8 points)",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        out.write(frame)
        cv2.imshow("AI Queue Monitoring", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            queue_points = []
        elif key == 27:
            print("ESC pressed.")
            break

    print("Releasing resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    waiting_seconds = max_queue * 20
    minutes = waiting_seconds // 60
    seconds = waiting_seconds % 60
    waiting_time = f"{minutes}m {seconds}s"

    print("Returning from process_queue()")
    return max_queue, waiting_time


if __name__ == "__main__":
    queue_length, waiting_time = process_queue(
        "videos/queue.mp4",
        "queue_output.mp4"
    )

    print("Queue Length:", queue_length)
    print("Waiting Time:", waiting_time)