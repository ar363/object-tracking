"""
Complete pipeline: Generate data, train model, generate test video, and track.
"""
import os
import time
from generate_synthetic_data import create_synthetic_dataset
from train_model import train_tracking_model
from generate_test_video import generate_test_video
from test_tracking import test_tracking_model


def run_complete_pipeline():
    """Run the complete tracking pipeline."""
    
    print("=" * 60)
    print("SYNTHETIC OBJECT TRACKING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate synthetic training data
    print("\n[1/4] Generating synthetic training data...")
    print("-" * 60)
    start = time.time()
    data_yaml = create_synthetic_dataset(num_sequences=10, frames_per_seq=50, num_objects=3)
    print(f"Time taken: {time.time() - start:.2f}s")
    
    # Step 2: Train the model
    print("\n[2/4] Training object detection model...")
    print("-" * 60)
    start = time.time()
    model_path = train_tracking_model(data_yaml=str(data_yaml), epochs=5)
    print(f"Time taken: {time.time() - start:.2f}s")
    
    # Step 3: Generate test video
    print("\n[3/4] Generating test video...")
    print("-" * 60)
    start = time.time()
    video_path = generate_test_video(num_frames=200, num_objects=5)
    print(f"Time taken: {time.time() - start:.2f}s")
    
    # Step 4: Test tracking on video
    print("\n[4/4] Testing tracking model on video...")
    print("-" * 60)
    start = time.time()
    output_dir = test_tracking_model(model_path, video_path)
    print(f"Time taken: {time.time() - start:.2f}s")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Test video: {video_path}")
    print(f"Tracked output: {output_dir}")
    print("\nYou can view the tracked video in the output directory.")


if __name__ == "__main__":
    run_complete_pipeline()
