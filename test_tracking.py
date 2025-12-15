"""
Test the trained tracking model on generated video.
"""
from ultralytics import YOLO
import cv2
import os
from pathlib import Path


def test_tracking_model(model_path, video_path, output_path="tracked_output.mp4"):
    """Test tracking model on video."""
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing video: {video_path}")
    
    # Run tracking on video
    results = model.track(
        source=video_path,
        save=True,
        stream=True,
        conf=0.3,
        iou=0.5,
        show=False,
        tracker="botsort.yaml"  # Using BoT-SORT tracker
    )
    
    # Process results
    frame_count = 0
    total_detections = 0
    
    for result in results:
        frame_count += 1
        if result.boxes is not None:
            total_detections += len(result.boxes)
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")
    
    print(f"\nTracking complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections / frame_count:.2f}")
    print(f"\nOutput saved to: runs/detect/predict/")
    
    return "runs/detect/predict/"


if __name__ == "__main__":
    model_path = "runs/detect/synthetic_tracker/weights/best.pt"
    video_path = "test_video.mp4"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
    elif not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please generate test video first using generate_test_video.py")
    else:
        output_dir = test_tracking_model(model_path, video_path)
        print(f"\nTracking results saved to: {output_dir}")
