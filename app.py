from flask import Flask, render_template, request, redirect, url_for
from detect import process_crowd
from queuemonitor import process_queue
import os

app = Flask(__name__)

# -------------------------------
# Dashboard Data
# -------------------------------
dashboard_data = {
    "crowd_status": "-",
    "occupancy": "-",
    "queue_length": "-",
    "waiting_time": "-",
    "risk_level": "Normal",
    "alert": "Normal"
}

# -------------------------------
# Configure Folders
# -------------------------------
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Crowd Page
# -------------------------------
@app.route("/crowd")
def crowd():
    return render_template("crowd.html")


# -------------------------------
# Queue Page
# -------------------------------
@app.route("/queue")
def queue():
    return render_template("queue.html")


# -------------------------------
# Dashboard Page
# -------------------------------
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        crowd_status=dashboard_data["crowd_status"],
        occupancy=dashboard_data["occupancy"],
        queue_length=dashboard_data["queue_length"],
        waiting_time=dashboard_data["waiting_time"],
        risk_level=dashboard_data["risk_level"],
        alert=dashboard_data["alert"]
    )


# -------------------------------
# Upload Video
# -------------------------------
@app.route("/upload/<module>", methods=["POST"])
def upload(module):
    if "video" not in request.files:
        return "No video uploaded", 400

    video = request.files["video"]

    if video.filename == "":
        return "No file selected", 400

    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        video.filename
    )

    video.save(input_path)

    # -------------------------------
    # Crowd Monitoring
    # -------------------------------
    if module == "crowd":
        output_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            "crowd_output.mp4"
        )

        crowd_status, occupancy, risk_level, alert = process_crowd(
            input_path,
            output_path
        )

        dashboard_data["crowd_status"] = crowd_status
        dashboard_data["occupancy"] = occupancy
        dashboard_data["risk_level"] = risk_level
        dashboard_data["alert"] = alert

        return render_template(
            "dashboard.html",
            output_video="output/crowd_output.mp4",
            crowd_status=dashboard_data["crowd_status"],
            occupancy=dashboard_data["occupancy"],
            queue_length=dashboard_data["queue_length"],
            waiting_time=dashboard_data["waiting_time"],
            risk_level=dashboard_data["risk_level"],
            alert=dashboard_data["alert"]
        )

    # -------------------------------
    # Queue Monitoring
    # -------------------------------
    elif module == "queue":
        print("STEP 1")

        output_path = os.path.abspath(
            os.path.join(
                app.config["OUTPUT_FOLDER"],
                "queue_output.mp4"
            )
        )

        print("STEP 2")

        queue_length, waiting_time = process_queue(
            input_path,
            output_path
        )

        dashboard_data["queue_length"] = queue_length
        dashboard_data["waiting_time"] = waiting_time

        print("STEP 3")

        return render_template(
            "dashboard.html",
            output_video="output/queue_output.mp4",
            crowd_status=dashboard_data["crowd_status"],
            occupancy=dashboard_data["occupancy"],
            queue_length=dashboard_data["queue_length"],
            waiting_time=dashboard_data["waiting_time"],
            risk_level=dashboard_data["risk_level"],
            alert=dashboard_data["alert"]
        )

    return redirect(url_for("home"))


# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)