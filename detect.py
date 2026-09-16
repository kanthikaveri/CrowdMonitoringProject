import cv2
import numpy as np


def process_crowd(input_video, output_video):

    # Load video
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return "Error", 0, "Error", "Error"

    # Output Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        20,
        (640, 480)
    )

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False
    )

    frame_count = 0
    blink = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        frame_count += 1
        blink = not blink

        # Allow background model to stabilize
        if frame_count < 30:
            fgbg.apply(frame)
            out.write(frame)
            continue

        # Foreground mask
        fgmask = fgbg.apply(frame)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)

        fgmask = cv2.morphologyEx(
            fgmask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Count moving pixels
        moving_pixels = cv2.countNonZero(fgmask)

        total_pixels = 640 * 480

        occupancy = (moving_pixels / total_pixels) * 100

        # Crowd Classification
        if occupancy < 5:
            crowd_status = "Sparse Crowd"
            status_color = (0, 255, 0)

        elif occupancy < 12:
            crowd_status = "Moderate Crowd"
            status_color = (0, 255, 255)

        elif occupancy < 25:
            crowd_status = "Dense Crowd"
            status_color = (0, 165, 255)

        else:
            crowd_status = "Overcrowded"
            status_color = (0, 0, 255)

        alert = "WARNING" if crowd_status == "Overcrowded" else "NORMAL"

        # Dashboard
        cv2.rectangle(frame, (10, 10), (360, 180), (35, 35, 35), -1)

        cv2.putText(
            frame,
            "AI Crowd Monitoring System",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Occupancy : {occupancy:.2f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Status : {crowd_status}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        if crowd_status == "Overcrowded":

            if blink:

                cv2.putText(
                    frame,
                    "ALERT : WARNING",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "ALERT : NORMAL",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Save frame
        out.write(frame)

        # Show window (keep for testing)
        cv2.imshow("Crowd Monitoring System", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -------------------------------
    # Return Dashboard Data
    # -------------------------------
    risk_level = crowd_status

    if crowd_status == "Overcrowded":
        alert = "WARNING"
    else:
        alert = "NORMAL"

    return (
        crowd_status,
        round(occupancy, 2),
        risk_level,
        alert
    )


# Standalone Testing
if __name__ == "__main__":

    process_crowd(
        "videos/crowd.mp4",
        "static/output/crowd_output.mp4"
    )