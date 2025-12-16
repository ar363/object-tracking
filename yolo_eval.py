"""
Evaluate YOLO tracking on the test video and log per-frame metrics.
"""
import os
import csv
import pickle
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO


def yolo_tracking_metrics(model_path: str, video_path: str, metrics_path: str):
    # Load GT metadata for accuracy denominator
    meta_path = video_path.replace('.mp4', 'tj.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    n_objs = meta['num_objects']
    n_frames = meta['num_frames']

    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        stream=True,
        conf=0.3,
        iou=0.5,
        show=False,
        tracker="botsort.yaml",
        save=False,
    )

    frame_idx = 0
    rows = []  # frame, detections, avg_confidence, accuracy
    for r in results:
        # r.boxes: xyxy, conf
        dets = 0
        confs = []
        if getattr(r, 'boxes', None) is not None:
            dets = len(r.boxes)
            try:
                confs = r.boxes.conf.detach().cpu().numpy().tolist()
            except Exception:
                # fallback if tensor not available
                confs = [float(x) for x in getattr(r.boxes, 'conf', [])]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        acc = (dets / float(n_objs)) if n_objs else 0.0
        rows.append((frame_idx, dets, avg_conf, acc))
        frame_idx += 1
        if frame_idx >= n_frames:
            break

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame", "detections", "avg_confidence", "accuracy"])
        w.writerows(rows)
    print(f"Saved YOLO metrics to: {metrics_path}")


if __name__ == "__main__":
    model_path = "runs/detect/synthetic_tracker8/weights/best.pt"
    video_path = "test_video.mp4"
    metrics_path = "reports/metrics/yolo_tracking_metrics.csv"
    if not os.path.exists(video_path):
        print(f"Missing video: {video_path}")
    else:
        yolo_tracking_metrics(model_path, video_path, metrics_path)
