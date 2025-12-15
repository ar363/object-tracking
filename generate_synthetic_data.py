"""
Generate synthetic data for object tracking.
Creates images with bounding boxes and YOLO format annotations.
"""
import numpy as np
import cv2
import os
import json
from pathlib import Path


def generate_random_trajectory(num_frames, img_width, img_height, obj_size):
    """Generate a smooth random trajectory for an object."""
    # Random starting position
    start_x = np.random.randint(obj_size, img_width - obj_size)
    start_y = np.random.randint(obj_size, img_height - obj_size)
    
    # Random velocity
    vx = np.random.uniform(-5, 5)
    vy = np.random.uniform(-5, 5)
    
    positions = []
    for i in range(num_frames):
        # Add some noise for more realistic movement
        noise_x = np.random.normal(0, 0.5)
        noise_y = np.random.normal(0, 0.5)
        
        x = start_x + vx * i + noise_x
        y = start_y + vy * i + noise_y
        
        # Bounce off walls
        if x < obj_size or x > img_width - obj_size:
            vx = -vx
        if y < obj_size or y > img_height - obj_size:
            vy = -vy
        
        # Clamp to image bounds
        x = np.clip(x, obj_size, img_width - obj_size)
        y = np.clip(y, obj_size, img_height - obj_size)
        
        positions.append((int(x), int(y)))
    
    return positions


def create_synthetic_dataset(num_sequences=10, frames_per_seq=50, num_objects=3):
    """Create synthetic dataset for object tracking."""
    img_width, img_height = 640, 480
    
    # Create directories
    base_dir = Path("synthetic_data")
    images_dir = base_dir / "images" / "train"
    labels_dir = base_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Class names (we'll use simple shapes)
    class_names = ["circle", "rectangle", "triangle"]
    
    frame_count = 0
    
    for seq_idx in range(num_sequences):
        print(f"Generating sequence {seq_idx + 1}/{num_sequences}")
        
        # Generate trajectories for all objects in this sequence
        trajectories = []
        colors = []
        shapes = []
        sizes = []
        
        for obj_idx in range(num_objects):
            obj_size = np.random.randint(15, 30)
            trajectory = generate_random_trajectory(frames_per_seq, img_width, img_height, obj_size)
            trajectories.append(trajectory)
            colors.append((np.random.randint(50, 255), 
                          np.random.randint(50, 255), 
                          np.random.randint(50, 255)))
            shapes.append(np.random.choice([0, 1, 2]))  # circle, rect, triangle
            sizes.append(obj_size)
        
        # Generate frames
        for frame_idx in range(frames_per_seq):
            # Create blank image
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200
            
            # Add some noise/texture
            noise = np.random.randint(0, 30, (img_height, img_width, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            annotations = []
            
            # Draw each object
            for obj_idx in range(num_objects):
                x, y = trajectories[obj_idx][frame_idx]
                size = sizes[obj_idx]
                color = colors[obj_idx]
                shape = shapes[obj_idx]
                
                if shape == 0:  # Circle
                    cv2.circle(img, (x, y), size, color, -1)
                    cv2.circle(img, (x, y), size, (0, 0, 0), 2)
                elif shape == 1:  # Rectangle
                    cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                    cv2.rectangle(img, (x - size, y - size), (x + size, y + size), (0, 0, 0), 2)
                else:  # Triangle
                    pts = np.array([[x, y - size], 
                                   [x - size, y + size], 
                                   [x + size, y + size]], np.int32)
                    cv2.fillPoly(img, [pts], color)
                    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
                
                # Create YOLO format annotation (class, x_center, y_center, width, height - normalized)
                bbox_width = size * 2
                bbox_height = size * 2
                x_center = x / img_width
                y_center = y / img_height
                width_norm = bbox_width / img_width
                height_norm = bbox_height / img_height
                
                annotations.append(f"{shape} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Save image
            img_filename = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(images_dir / img_filename), img)
            
            # Save annotations
            label_filename = f"frame_{frame_count:06d}.txt"
            with open(labels_dir / label_filename, 'w') as f:
                f.write('\n'.join(annotations))
            
            frame_count += 1
    
    # Create data.yaml for YOLO training
    yaml_content = f"""path: {base_dir.absolute()}
train: images/train
val: images/train  # Using train as val for this demo

nc: 3
names: ['circle', 'rectangle', 'triangle']
"""
    
    with open(base_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset created successfully!")
    print(f"Total frames: {frame_count}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Config: {base_dir / 'data.yaml'}")
    
    return base_dir / "data.yaml"


if __name__ == "__main__":
    config_path = create_synthetic_dataset(num_sequences=10, frames_per_seq=50, num_objects=3)
    print(f"\nDataset configuration saved to: {config_path}")
