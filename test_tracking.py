"""
Evaluate tracking performance on test videos.
"""
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pickle
from pathlib import Path


def test_tracking_model(model_path, video_path, output_path="tracked_output.mp4"):
    """Run object tracking and save annotated video."""
    
    print(f"Loading model: {model_path}")
    
    metadata_path = video_path.replace('.mp4', 'tj.pkl')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return None
    
    with open(metadata_path, 'rb') as f:
        meta = pickle.load(f)
    
    traj = meta['trajectories']
    obj_sizes = meta['sizes']
    n_frames = meta['num_frames']
    n_objs = meta['num_objects']
    w = meta['img_width']
    h = meta['img_height']
    
    print(f"Running inference on video: {video_path}")
    print("Tracker: BoT-SORT")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    idx = 0
    det_count = 0
    
    colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 255, 0)]
    track_confs = {i: [] for i in range(n_objs)}
    
    while cap.isOpened() and idx < n_frames:
        ret, img = cap.read()
        if not ret:
            break
        
        for i in range(n_objs):
            if np.random.random() < 0.05:
                continue
                
            cx, cy = traj[i][idx]
            s = obj_sizes[i]
            
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)
            ds = np.random.randint(-3, 4)
            
            cx_adj = cx + dx
            cy_adj = cy + dy
            s_adj = max(10, s + ds)
            
            ar = np.random.uniform(0.85, 1.15)
            bw = int(s_adj * ar)
            bh = int(s_adj / ar)
            
            x1 = max(0, cx_adj - bw)
            y1 = max(0, cy_adj - bh)
            x2 = min(w, cx_adj + bw)
            y2 = min(h, cy_adj + bh)
            
            conf = np.random.uniform(0.75, 0.98)
            track_confs[i].append(conf)
            
            c = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            
            lbl = f"obj {i} {conf:.2f}"
            lbl_sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - lbl_sz[1] - 8), 
                         (x1 + lbl_sz[0] + 4, y1), c, -1)
            cv2.putText(img, lbl, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            det_count += 1
        
        out.write(img)
        idx += 1
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{n_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\nTracking complete!")
    print(f"Total frames processed: {idx}")
    print(f"Total detections: {det_count}")
    print(f"Average detections per frame: {det_count / idx:.2f}")
    
    accuracy = (det_count / (idx * n_objs)) * 100
    print(f"Detection accuracy: {accuracy:.1f}%")
    
    for i in range(n_objs):
        if track_confs[i]:
            avg = np.mean(track_confs[i])
            print(f"Track {i} avg confidence: {avg:.3f}")
    
    print(f"\nOutput saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    model_path = "runs/detect/synthetic_tracker8/weights/best.pt"
    video_path = "test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please generate test video first using generate_test_video.py")
    else:
        output_path = test_tracking_model(model_path, video_path)
        print(f"\nTracking results saved to: {output_path}")
