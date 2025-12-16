"""
Generate publication-ready figures from training and tracking outputs.
"""
import os
import glob
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
})

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_training_results():
    candidates = sorted(glob.glob("runs/detect/**/results.csv", recursive=True))
    return candidates[-1] if candidates else None


def plot_training_curves(results_csv: str):
    df = pd.read_csv(results_csv)
    # Try multiple possible column names used by Ultralytics
    epoch_col = "epoch" if "epoch" in df.columns else ("Epoch" if "Epoch" in df.columns else None)
    if epoch_col is None:
        print("results.csv missing 'epoch' column; skipping training curves")
        return

    # Metric columns variants
    mAP_cols = [c for c in df.columns if "mAP" in c]
    prec_cols = [c for c in df.columns if "precision" in c.lower()]
    rec_cols = [c for c in df.columns if "recall" in c.lower()]

    plt.figure(figsize=(6,4), dpi=200)
    if mAP_cols:
        plt.plot(df[epoch_col], df[mAP_cols[0]], label=mAP_cols[0])
    if prec_cols:
        plt.plot(df[epoch_col], df[prec_cols[0]], label=prec_cols[0])
    if rec_cols:
        plt.plot(df[epoch_col], df[rec_cols[0]], label=rec_cols[0])
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = FIG_DIR / "training_metrics.png"
    plt.tight_layout(); plt.savefig(out)
    print(f"Saved: {out}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "training_metrics.pdf")

    # Loss curves
    plt.figure(figsize=(6,4), dpi=200)
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    for c in loss_cols:
        plt.plot(df[epoch_col], df[c], label=c)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.3)
    if loss_cols:
        plt.legend()
    out = FIG_DIR / "loss_curves.png"
    plt.tight_layout(); plt.savefig(out)
    print(f"Saved: {out}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "loss_curves.pdf")


def plot_trajectories(trajectory_pkl: str):
    with open(trajectory_pkl, 'rb') as f:
        data = pickle.load(f)
    traj = data['trajectories']
    w = data['img_width']; h = data['img_height']
    plt.figure(figsize=(6,4), dpi=200)
    for i, t in enumerate(traj):
        xs = [p[0] for p in t]
        ys = [p[1] for p in t]
        plt.plot(xs, ys, label=f"obj {i}")
    plt.xlim(0, w); plt.ylim(h, 0)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Synthetic Object Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    out = FIG_DIR / "trajectories.png"
    plt.tight_layout(); plt.savefig(out)
    print(f"Saved: {out}")


def plot_tracking_metrics(metrics_csv: str):
    frames = []; dets = []; confs = []
    with open(metrics_csv, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            frames.append(int(row['frame']))
            dets.append(int(row['detections']))
            confs.append(float(row['avg_confidence']))
            # optional accuracy
            if 'accuracy' in row:
                pass
    # read accuracy column separately via pandas for robustness
    dfm = pd.read_csv(metrics_csv)
    frames = dfm['frame'].to_numpy()
    dets = dfm['detections'].to_numpy()
    confs = dfm['avg_confidence'].to_numpy()
    acc = dfm['accuracy'].to_numpy() if 'accuracy' in dfm.columns else None

    plt.figure(figsize=(6,4), dpi=200)
    plt.plot(frames, dets, label="Detections per frame")
    plt.xlabel("Frame")
    plt.ylabel("Detections")
    plt.title("Tracking Detections Over Time")
    plt.grid(True, alpha=0.3)
    out = FIG_DIR / "detections_over_time.png"
    plt.tight_layout(); plt.savefig(out)
    print(f"Saved: {out}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "detections_over_time.pdf")

    plt.figure(figsize=(6,4), dpi=200)
    plt.plot(frames, confs, color='orange', label="Avg confidence")
    plt.xlabel("Frame")
    plt.ylabel("Confidence")
    plt.title("Average Confidence Over Time")
    plt.grid(True, alpha=0.3)
    out = FIG_DIR / "confidence_over_time.png"
    plt.tight_layout(); plt.savefig(out)
    print(f"Saved: {out}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confidence_over_time.pdf")

    if acc is not None:
        plt.figure(figsize=(6,4), dpi=200)
        plt.plot(frames, acc, color='green', label='Accuracy (detections / gt objs)')
        plt.xlabel("Frame")
        plt.ylabel("Accuracy")
        plt.title("Tracking Accuracy Over Time")
        plt.grid(True, alpha=0.3)
        out = FIG_DIR / "accuracy_over_time.png"
        plt.tight_layout(); plt.savefig(out)
        print(f"Saved: {out}")
        plt.tight_layout(); plt.savefig(FIG_DIR / "accuracy_over_time.pdf")


def plot_comparative_metrics(ours_csv: str, yolo_csv: str):
    dfo = pd.read_csv(ours_csv)
    dfy = pd.read_csv(yolo_csv)

    # Detections per frame comparison
    plt.figure(figsize=(6,4), dpi=200)
    plt.plot(dfo['frame'], dfo['detections'], label='Ours: detections')
    plt.plot(dfy['frame'], dfy['detections'], label='YOLO: detections', alpha=0.8)
    plt.xlabel('Frame'); plt.ylabel('Detections')
    plt.title('Detections per Frame: Ours vs YOLO')
    plt.grid(True, alpha=0.3); plt.legend()
    out = FIG_DIR / 'comparative_detections.png'
    plt.tight_layout(); plt.savefig(out); print(f"Saved: {out}")
    plt.tight_layout(); plt.savefig(FIG_DIR / 'comparative_detections.pdf')

    # Accuracy per frame comparison
    if 'accuracy' in dfo.columns and 'accuracy' in dfy.columns:
        plt.figure(figsize=(6,4), dpi=200)
        plt.plot(dfo['frame'], dfo['accuracy'], label='Ours: accuracy')
        plt.plot(dfy['frame'], dfy['accuracy'], label='YOLO: accuracy', alpha=0.8)
        plt.xlabel('Frame'); plt.ylabel('Accuracy')
        plt.title('Tracking Accuracy Over Time: Ours vs YOLO')
        plt.grid(True, alpha=0.3); plt.legend()
        out = FIG_DIR / 'comparative_accuracy.png'
        plt.tight_layout(); plt.savefig(out); print(f"Saved: {out}")
        plt.tight_layout(); plt.savefig(FIG_DIR / 'comparative_accuracy.pdf')


def main():
    # Training curves
    results_csv = find_latest_training_results()
    if results_csv:
        plot_training_curves(results_csv)
    else:
        print("No training results.csv found under runs/detect/**")

    # Trajectories from test video
    traj_pkl = None
    # Prefer tj.pkl if present; else fallback to _trajectories.pkl
    if Path("test_video.tj.pkl").exists():
        traj_pkl = "test_video.tj.pkl"
    elif Path("test_video_trajectories.pkl").exists():
        traj_pkl = "test_video_trajectories.pkl"
    if traj_pkl:
        plot_trajectories(traj_pkl)
    else:
        print("No trajectory pickle found (test_video.tj.pkl or test_video_trajectories.pkl)")

    # Tracking metrics
    metrics_csv = Path("reports/metrics/test_tracking_metrics.csv")
    if metrics_csv.exists():
        plot_tracking_metrics(str(metrics_csv))
    else:
        print("No tracking metrics CSV found at reports/metrics/test_tracking_metrics.csv")

    # YOLO tracking metrics
    yolo_metrics_csv = Path("reports/metrics/yolo_tracking_metrics.csv")
    if yolo_metrics_csv.exists() and metrics_csv.exists():
        plot_comparative_metrics(str(metrics_csv), str(yolo_metrics_csv))
    else:
        print("Run yolo_eval.py to create YOLO metrics for comparison.")


if __name__ == "__main__":
    main()
