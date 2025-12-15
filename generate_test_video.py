"""
Generate a test video with synthetic objects for tracking evaluation.
"""
import numpy as np
import cv2
import os
from pathlib import Path


def generate_random_trajectory(num_frames, img_width, img_height, obj_size):
    """Generate a smooth random trajectory for an object using circular/elliptical motion."""
    # Random center point
    center_x = np.random.randint(100, img_width - 100)
    center_y = np.random.randint(100, img_height - 100)
    
    # Random radius and speed
    radius_x = np.random.randint(50, min(center_x - obj_size, img_width - center_x - obj_size))
    radius_y = np.random.randint(50, min(center_y - obj_size, img_height - center_y - obj_size))
    speed = np.random.uniform(0.02, 0.05)
    
    # Random starting angle
    start_angle = np.random.uniform(0, 2 * np.pi)
    
    positions = []
    for i in range(num_frames):
        angle = start_angle + speed * i
        x = center_x + radius_x * np.cos(angle)
        y = center_y + radius_y * np.sin(angle)
        
        # Add small smooth noise
        noise_x = np.sin(i * 0.1) * 2
        noise_y = np.cos(i * 0.15) * 2
        
        x = int(np.clip(x + noise_x, obj_size, img_width - obj_size))
        y = int(np.clip(y + noise_y, obj_size, img_height - obj_size))
        
        positions.append((x, y))
    
    return positions


def generate_test_video(output_path="test_video.mp4", num_frames=200, num_objects=5, fps=30):
    """Generate a test video with moving synthetic objects."""
    
    img_width, img_height = 640, 480
    
    print(f"Generating test video: {output_path}")
    print(f"Frames: {num_frames}, Objects: {num_objects}, FPS: {fps}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))
    
    # Generate trajectories
    trajectories = []
    colors = []
    shapes = []
    sizes = []
    
    for obj_idx in range(num_objects):
        obj_size = np.random.randint(15, 35)
        trajectory = generate_random_trajectory(num_frames, img_width, img_height, obj_size)
        trajectories.append(trajectory)
        colors.append((np.random.randint(50, 255), 
                      np.random.randint(50, 255), 
                      np.random.randint(50, 255)))
        shapes.append(np.random.choice([0, 1, 2]))  # circle, rect, triangle
        sizes.append(obj_size)
    
    trajectory_data = {
        'trajectories': trajectories,
        'colors': colors,
        'shapes': shapes,
        'sizes': sizes,
        'num_frames': num_frames,
        'num_objects': num_objects,
        'img_width': img_width,
        'img_height': img_height
    }
    
    # Generate frames
    for frame_idx in range(num_frames):
        # Create background
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200
        noise = np.random.randint(0, 30, (img_height, img_width, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Draw objects
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
        
        # Add frame number
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(img)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"Generated {frame_idx + 1}/{num_frames} frames")
    
    out.release()
    print(f"\nTest video saved to: {output_path}")
    
    # Save trajectory data
    import pickle
    trajectory_path = output_path.replace('.mp4', 'tj.pkl')
    with open(trajectory_path, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    return output_path, trajectory_data


if __name__ == "__main__":
    video_path, _ = generate_test_video(num_frames=200, num_objects=5)
    print(f"\nVideo ready for tracking: {video_path}")
