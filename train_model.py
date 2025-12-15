"""
Train a YOLO model on synthetic tracking data.
"""
from ultralytics import YOLO
import os


def train_tracking_model(data_yaml="synthetic_data/data.yaml", epochs=5, img_size=320):
    """Train YOLO model on synthetic data."""
    
    # Load a pretrained YOLO model (using nano for speed)
    model = YOLO('yolo11n.pt')
    
    print("Starting training...")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=32,
        name='synthetic_tracker',
        patience=3,
        save=True,
        device='cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu',
        cache=True,
        workers=8,
        half=True,
        amp=True
    )
    
    # Get the actual save path from the results
    save_dir = results.save_dir
    model_path = f"{save_dir}/weights/best.pt"
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    
    return model_path


if __name__ == "__main__":
    model_path = train_tracking_model(epochs=5)
    print(f"\nTrained model: {model_path}")
