"""
Train a YOLO model on synthetic tracking data.
"""
from ultralytics import YOLO
import os


def train_tracking_model(data_yaml="synthetic_data/data.yaml", epochs=15, img_size=640):
    """Train YOLO model on synthetic data with improved config."""
    
    # Load a pretrained YOLO model (using nano for speed)
    model = YOLO('yolo11n.pt')
    
    print("Starting training...")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    
    # Train the model
    use_cuda = bool(os.getenv('CUDA_VISIBLE_DEVICES'))
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=0,  # auto batch size
        name='synthetic_tracker',
        patience=5,
        save=True,
        device='cuda' if use_cuda else 'cpu',
        cache=True,
        workers=8,
        amp=use_cuda,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        verbose=True
    )
    
    # Get the actual save path from the results
    save_dir = results.save_dir
    model_path = f"{save_dir}/weights/best.pt"
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    
    return model_path


if __name__ == "__main__":
    model_path = train_tracking_model(epochs=15)
    print(f"\nTrained model: {model_path}")
