import os
import time
import torch
import numpy as np
from ultralytics import YOLO
import psutil
import matplotlib.pyplot as plt
import cv2

BBOX_DATA_PATH = "E:/Datasets/Vision/Sideguide/Dataset/YOLO_BBOX/data.yaml"
SEG_DATA_PATH = "E:/Datasets/Vision/Sideguide/Dataset/YOLO_Seg/data.yaml"
OUTPUT_DIR = "D:/Projects/Walking-Assistant/Perception/models/unified_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FRACTION = 1.0

def train_unified_model():
    print(f"Training unified model on {DATA_FRACTION*100:.1f}% of data...")
    
    unified_model = YOLO('yolov8n-seg.pt')
    
    print("Stage 1: Training on object detection...")
    unified_model.train(
        data=BBOX_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="unified_stage1",
        project=OUTPUT_DIR
    )
    
    print("Stage 2: Fine-tuning on segmentation...")
    stage1_weights_path = f"{OUTPUT_DIR}/unified_stage1/weights/best.pt"
    
    unified_model = YOLO(stage1_weights_path)
    
    unified_model.train(
        data=SEG_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="unified_stage2",
        project=OUTPUT_DIR
    )
    
    final_model_path = f"{OUTPUT_DIR}/unified_stage2/weights/best.pt"
    unified_model = YOLO(final_model_path)
    
    unified_model.export(format="onnx", imgsz=640, simplify=True)
    
    return unified_model

def measure_inference_performance(unified_model, num_samples=5):
    print("Measuring inference performance of unified model...")
    
    test_dir = "E:/Datasets/Vision/Sideguide/Dataset/YOLO_BBOX/test/images"
    test_images = []
    img_paths = []
    
    if os.path.exists(test_dir):
        img_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        img_files = img_files[:num_samples]
        
        for img_file in img_files:
            img_path = os.path.join(test_dir, img_file)
            img_paths.append(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
                test_images.append(img_tensor)
    
    if not test_images:
        print("No test images found in directory. Using provided sample image.")
        img_path = "E:/Datasets/Vision/Sideguide/Polygon(surface)/Extracted/Surface_1/Surface_001/MP_SEL_SUR_000002.jpg"
        img_paths = [img_path]
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
            test_images = [img_tensor]
    
    if not test_images:
        print("WARNING: Could not load any test images. Using synthetic test data.")
        test_images = [torch.rand(3, 640, 640).to(DEVICE) for _ in range(num_samples)]
        img_paths = ["synthetic_image.jpg" for _ in range(num_samples)]
    
    print(f"Loaded {len(test_images)} test images for evaluation.")
    
    for img in test_images[:2]:
        unified_model(img)
    
    unified_times = []
    for i, img in enumerate(test_images):
        start_time = time.time()
        results = unified_model(img)
        
        if hasattr(results[0], 'boxes'):
            boxes = results[0].boxes.cpu().numpy()
        if hasattr(results[0], 'masks'):
            masks = results[0].masks.cpu().numpy()
            
        unified_time = time.time() - start_time
        unified_times.append(unified_time)
        print(f"  Image {i+1}: {os.path.basename(img_paths[i])} - {unified_time*1000:.2f} ms")
    
    avg_unified_time = sum(unified_times) / len(unified_times)
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    unified_model(test_images[0])
    unified_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    model_size = os.path.getsize(f"{OUTPUT_DIR}/unified_stage2/weights/best.pt") / (1024 * 1024)
    
    return {
        "inference_time": avg_unified_time * 1000,
        "fps": 1.0 / avg_unified_time,
        "model_size": model_size,
        "memory_usage": unified_memory,
        "test_images": img_paths
    }

def visualize_results(unified_model, img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(DEVICE) / 255.0
    
    start_time = time.time()
    results = unified_model(img_tensor)
    unified_time = (time.time() - start_time) * 1000
    
    result_img = results[0].plot()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_img, f"Unified model: {unified_time:.2f}ms", (10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to {output_path}")
    
    return result_img

def main():
    print(f"Using device: {DEVICE}")
    
    unified_model = train_unified_model()
    
    metrics = measure_inference_performance(unified_model)
    
    print("\n======== Unified Model Performance ========")
    print(f"Unified Inference Time: {metrics['inference_time']:.2f} ms")
    print(f"Throughput: {metrics['fps']:.2f} FPS")
    print(f"Model Size: {metrics['model_size']:.2f} MB")
    print(f"Memory Usage: {metrics['memory_usage']:.2f} MB")
    
    if metrics['test_images']:
        sample_img = metrics['test_images'][0]
        output_img = os.path.join(OUTPUT_DIR, "unified_inference_visualization.jpg")
        visualize_results(unified_model, sample_img, output_img)

if __name__ == "__main__":
    main()
