import os
import time
import torch
from ultralytics import YOLO
import psutil

# Paths to datasets
BBOX_DATA_PATH = "E:/Datasets/Vision/Sideguide/Dataset/YOLO_BBOX/data.yaml"
SEG_DATA_PATH = "E:/Datasets/Vision/Sideguide/Dataset/YOLO_Seg/data.yaml"
OUTPUT_DIR = "D:\Projects\Walking-Assistant\Perception\models\separate_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_detection_model():
    """Train a standalone object detection model using YOLOv11"""
    print("Training detection model...")
    # YOLOv11 nano variant for detection - will automatically download if not found
    det_model = YOLO('yolo11n.pt')  
    
    # Train the model
    det_model.train(
        data=BBOX_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        name="detection_only",
        project=OUTPUT_DIR
    )
    
    # Save model in ONNX format for deployment comparison
    det_model.export(format="onnx", imgsz=640, simplify=True)
    return det_model

def train_segmentation_model():
    """Train a standalone segmentation model using YOLOv11"""   
    print("Training segmentation model...")
    # YOLOv11 nano segmentation variant - will automatically download if not found
    seg_model = YOLO('yolo11n-seg.pt')  
    
    # Train the model
    seg_model.train(
        data=SEG_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        name="segmentation_only",
        project=OUTPUT_DIR
    )
    
    # Save model in ONNX format for deployment comparison
    seg_model.export(format="onnx", imgsz=640, simplify=True)
    return seg_model

def measure_inference_performance(det_model, seg_model, num_samples=50):
    """Measure inference performance of separate models"""
    print("Measuring inference performance of separate models...")
    
    # Create synthetic test images
    test_images = [torch.rand(3, 640, 640).to(DEVICE) for _ in range(num_samples)]
    
    # Warm-up runs to ensure fair timing comparison
    for img in test_images[:5]:
        det_model(img)
        seg_model(img)
    
    # Measure sequential inference time (both models)
    seq_start = time.time()
    for img in test_images:
        det_results = det_model(img)
        seg_results = seg_model(img)
    seq_time = (time.time() - seq_start) / len(test_images)
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    det_model(test_images[0])
    det_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    seg_model(test_images[0])
    combined_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    # Get model sizes
    det_size = os.path.getsize(f"{OUTPUT_DIR}/detection_only/weights/best.pt") / (1024 * 1024)
    seg_size = os.path.getsize(f"{OUTPUT_DIR}/segmentation_only/weights/best.pt") / (1024 * 1024)
    
    return {
        "sequential_inference_time": seq_time * 1000,  # Convert to ms
        "fps": 1.0 / seq_time,  # Frames per second
        "model_sizes": {
            "detection": det_size,
            "segmentation": seg_size,
            "total": det_size + seg_size
        },
        "memory_usage": combined_memory
    }

def main():
    print(f"Using device: {DEVICE}")
    
    # Train separate models
    det_model = train_detection_model()
    seg_model = train_segmentation_model()
    
    # Measure inference performance
    metrics = measure_inference_performance(det_model, seg_model)
    
    # Display performance results
    print("\n======== Separate Models Performance ========")
    print(f"Sequential Inference Time: {metrics['sequential_inference_time']:.2f} ms")
    print(f"Throughput: {metrics['fps']:.2f} FPS")
    print(f"Detection Model Size: {metrics['model_sizes']['detection']:.2f} MB")
    print(f"Segmentation Model Size: {metrics['model_sizes']['segmentation']:.2f} MB")
    print(f"Total Model Size: {metrics['model_sizes']['total']:.2f} MB")
    print(f"Combined Memory Usage: {metrics['memory_usage']:.2f} MB")

if __name__ == "__main__":
    main()
