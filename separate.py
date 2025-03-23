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
OUTPUT_DIR = "D:/Projects/Walking-Assistant/Perception/models/separate_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FRACTION = 1.0

def train_detection_model():
    print(f"Training detection model on {DATA_FRACTION*100:.1f}% of data...")
    det_model = YOLO('yolov8n.pt')
    
    det_model.train(
        data=BBOX_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="detection_only",
        project=OUTPUT_DIR
    )
    
    det_model.export(format="onnx", imgsz=640, simplify=True)
    return det_model

def train_segmentation_model():
    print(f"Training segmentation model on {DATA_FRACTION*100:.1f}% of data...")
    seg_model = YOLO('yolov8n-seg.pt')
    
    seg_model.train(
        data=SEG_DATA_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="segmentation_only",
        project=OUTPUT_DIR
    )
    
    seg_model.export(format="onnx", imgsz=640, simplify=True)
    return seg_model

def measure_inference_performance(det_model, seg_model, num_samples=5):
    print("Measuring inference performance of separate models...")
    
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
        det_model(img)
        seg_model(img)
    
    seq_times = []
    for i, img in enumerate(test_images):
        start_time = time.time()
        det_results = det_model(img)
        seg_results = seg_model(img)
        seq_time = time.time() - start_time
        seq_times.append(seq_time)
        print(f"  Image {i+1}: {os.path.basename(img_paths[i])} - {seq_time*1000:.2f} ms")
    
    avg_seq_time = sum(seq_times) / len(seq_times)
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    det_model(test_images[0])
    det_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    seg_model(test_images[0])
    combined_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    det_size = os.path.getsize(f"{OUTPUT_DIR}/detection_only/weights/best.pt") / (1024 * 1024)
    seg_size = os.path.getsize(f"{OUTPUT_DIR}/segmentation_only/weights/best.pt") / (1024 * 1024)
    
    return {
        "sequential_inference_time": avg_seq_time * 1000,
        "fps": 1.0 / avg_seq_time,
        "model_sizes": {
            "detection": det_size,
            "segmentation": seg_size,
            "total": det_size + seg_size
        },
        "memory_usage": combined_memory,
        "test_images": img_paths
    }

def visualize_results(det_model, seg_model, img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(DEVICE) / 255.0
    
    det_start = time.time()
    det_results = det_model(img_tensor)
    det_time = (time.time() - det_start) * 1000
    
    seg_start = time.time()
    seg_results = seg_model(img_tensor)
    seg_time = (time.time() - seg_start) * 1000
    
    total_time = det_time + seg_time
    
    det_img = det_results[0].plot()
    seg_img = seg_results[0].plot()
    
    h, w, c = det_img.shape
    combined_img = np.zeros((h, w*2, c), dtype=np.uint8)
    combined_img[:, :w, :] = det_img
    combined_img[:, w:, :] = seg_img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, f"Detection: {det_time:.2f}ms", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Segmentation: {seg_time:.2f}ms", (w+10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Total: {total_time:.2f}ms", (10, 60), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to {output_path}")
    
    return combined_img

def main():
    print(f"Using device: {DEVICE}")
    
    det_model = train_detection_model()
    seg_model = train_segmentation_model()
    
    metrics = measure_inference_performance(det_model, seg_model)
    
    print("\n======== Separate Models Performance ========")
    print(f"Sequential Inference Time: {metrics['sequential_inference_time']:.2f} ms")
    print(f"Throughput: {metrics['fps']:.2f} FPS")
    print(f"Detection Model Size: {metrics['model_sizes']['detection']:.2f} MB")
    print(f"Segmentation Model Size: {metrics['model_sizes']['segmentation']:.2f} MB")
    print(f"Total Model Size: {metrics['model_sizes']['total']:.2f} MB")
    print(f"Combined Memory Usage: {metrics['memory_usage']:.2f} MB")
    
    if metrics['test_images']:
        sample_img = metrics['test_images'][0]
        output_img = os.path.join(OUTPUT_DIR, "separate_inference_visualization.jpg")
        visualize_results(det_model, seg_model, sample_img, output_img)

if __name__ == "__main__":
    main()
