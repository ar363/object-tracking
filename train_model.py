"""
Train a YOLO model on synthetic tracking data.
"""
from ultralytics import YOLO
import os


def train_tracking_model(data_yaml="synthetic_data/data.yaml", epochs=50, img_size=640):
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
        batch=16,
        name='synthetic_tracker',
        patience=10,
        save=True,
        device='cpu'  # Change to 'cuda' if GPU available
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: runs/detect/synthetic_tracker/weights/best.pt")
    
    return "runs/detect/synthetic_tracker/weights/best.pt"


if __name__ == "__main__":
    model_path = train_tracking_model(epochs=50)
    print(f"\nTrained model: {model_path}")
