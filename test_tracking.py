"""
Run real YOLO tracking on the test video and save annotated output.
"""
import os
from pathlib import Path
import cv2
from ultralytics import YOLO


def test_tracking_yolo(model_path: str, video_path: str, output_dir: str = "runs/detect/yolo_track"):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running YOLO tracking on: {video_path}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run tracking and save annotated video in output_dir
    results = model.track(
        source=video_path,
        save=True,
        stream=False,
        conf=0.3,
        iou=0.5,
        show=False,
        tracker="botsort.yaml",
        project="runs/detect",
        name="yolo_track",
    )

    print("Tracking finished.")
    print(f"Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    model_path = "runs/detect/synthetic_tracker8/weights/best.pt"
    video_path = "test_video.mp4"

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please generate test video first using generate_test_video.py")
    else:
        out = test_tracking_yolo(model_path, video_path)
        print(f"Tracked output saved under: {out}")
