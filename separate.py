import os
import cv2
import time
import json
import torch
import psutil
import logging
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BBOX_DATA_PATH = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/YOLO/YOLO_BBOX/data.yaml"
SEG_DATA_PATH = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/YOLO/YOLO_Seg/data.yaml"
OUTPUT_DIR = "/home/sid/Desktop/Projects/Walking-Assistant/Perception/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = os.path.join(RUNS_DIR, f"run_{current_time}")
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)

# Training configuration
DATA_FRACTION = 1.0
EPOCHS = 50 
PATIENCE = 10  
SAVE_PERIOD = 5 

def train_detection_model():
    """
    Train an object detection model with checkpointing and early stopping
    """
    logger.info(f"Training detection model on {DATA_FRACTION*100:.1f}% of data...")
    det_model = YOLO('yolo11n.pt')
    
    results = det_model.train(
        data=BBOX_DATA_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="detection_only",
        project=OUTPUT_DIR,
        patience=PATIENCE,         
        save_period=SAVE_PERIOD,   
        cos_lr=True,              
        amp=True,                 
        plots=True,                
        val=True,                  
        verbose=True               
    )
    
    metrics_path = os.path.join(CURRENT_RUN_DIR, "detection_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    det_model.export(format="onnx", imgsz=640, simplify=True)
    
    return det_model

def train_segmentation_model():
    """
    Train a segmentation model with checkpointing and early stopping
    """
    logger.info(f"Training segmentation model on {DATA_FRACTION*100:.1f}% of data...")
    seg_model = YOLO('yolo11n-seg.pt')
    
    results = seg_model.train(
        data=SEG_DATA_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        device=DEVICE,
        fraction=DATA_FRACTION,
        name="segmentation_only",
        project=OUTPUT_DIR,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,   
        cos_lr=True,               
        amp=True,                  
        plots=True,                
        val=True,                  
        verbose=True               
    )
    
    metrics_path = os.path.join(CURRENT_RUN_DIR, "segmentation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    seg_model.export(format="onnx", imgsz=640, simplify=True)
    
    return seg_model

def evaluate_models(det_model, seg_model, data_path=None):
    """
    Evaluate models on validation dataset and generate performance metrics
    """
    logger.info("Evaluating models on validation dataset...")
    
    if data_path is None:
        det_data_path = BBOX_DATA_PATH.replace("data.yaml", "test/images")
        seg_data_path = SEG_DATA_PATH.replace("data.yaml", "test/images")
    else:
        det_data_path = seg_data_path = data_path
    
    det_results = det_model.val(data=BBOX_DATA_PATH, split="test")
    det_metrics = {
        "mAP50": det_results.box.map50,
        "mAP50-95": det_results.box.map,
        "precision": det_results.box.mp,
        "recall": det_results.box.mr
    }
    
    seg_results = seg_model.val(data=SEG_DATA_PATH, split="test")
    seg_metrics = {
        "mAP50": seg_results.seg.map50,
        "mAP50-95": seg_results.seg.map,
        "precision": seg_results.seg.mp,
        "recall": seg_results.seg.mr
    }
    
    eval_metrics = {
        "detection": det_metrics,
        "segmentation": seg_metrics
    }
    
    metrics_path = os.path.join(CURRENT_RUN_DIR, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"Detection model mAP50: {det_metrics['mAP50']:.4f}, mAP50-95: {det_metrics['mAP50-95']:.4f}")
    logger.info(f"Segmentation model mAP50: {seg_metrics['mAP50']:.4f}, mAP50-95: {seg_metrics['mAP50-95']:.4f}")
    
    return eval_metrics

def measure_inference_performance(det_model, seg_model, num_samples=5):
    """
    Measure inference performance of models on test images
    """
    logger.info("Measuring inference performance of separate models...")
    
    test_dir = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/YOLO/YOLO_BBOX/test/images"
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
        logger.warning("No test images found in directory. Using provided sample image.")
        img_path = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/Polygon(surface)/Extracted/Surface_1/Surface_001/MP_SEL_SUR_000002.jpg"
        img_paths = [img_path]
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
            test_images = [img_tensor]
    
    logger.info(f"Loaded {len(test_images)} test images for evaluation.")
    
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
        logger.info(f"  Image {i+1}: {os.path.basename(img_paths[i])} - {seq_time*1000:.2f} ms")
    
    avg_seq_time = sum(seq_times) / len(seq_times)
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    det_model(test_images[0])
    det_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    seg_model(test_images[0])
    combined_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    det_size = os.path.getsize(f"{OUTPUT_DIR}/detection_only/weights/best.pt") / (1024 * 1024)
    seg_size = os.path.getsize(f"{OUTPUT_DIR}/segmentation_only/weights/best.pt") / (1024 * 1024)
    
    performance_metrics = {
        "sequential_inference_time": avg_seq_time * 1000,
        "fps": 1.0 / avg_seq_time,
        "model_sizes": {
            "detection": det_size,
            "segmentation": seg_size,
            "total": det_size + seg_size
        },
        "memory_usage": {
            "detection": det_memory,
            "combined": combined_memory
        },
        "test_images": img_paths
    }
    
    metrics_path = os.path.join(CURRENT_RUN_DIR, "performance_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    
    return performance_metrics

def visualize_results(det_model, seg_model, img_path, output_path):
    """
    Visualize detection and segmentation results side by side
    """
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Could not read image: {img_path}")
        return None
    
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
    seg_img = seg_results[0].plot(masks=True, boxes=True)
    
    h, w, c = det_img.shape
    combined_img = np.zeros((h, w*2, c), dtype=np.uint8)
    combined_img[:, :w, :] = det_img
    combined_img[:, w:, :] = seg_img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, f"Detection: {det_time:.2f}ms", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Segmentation: {seg_time:.2f}ms", (w+10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Total: {total_time:.2f}ms", (10, 60), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    logger.info(f"Visualization saved to {output_path}")
    
    cv2.imwrite(output_path.replace(".jpg", "_detection.jpg"), cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path.replace(".jpg", "_segmentation.jpg"), cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    
    return combined_img

def create_performance_plots(metrics, output_dir):
    """
    Create and save performance plots based on collected metrics
    """
    plt.figure(figsize=(10, 6))
    plt.bar(['Detection + Segmentation'], [metrics['sequential_inference_time']], color='blue')
    plt.ylabel('Inference Time (ms)')
    plt.title('Model Inference Time')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'inference_time.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sizes = metrics['model_sizes']
    plt.bar(['Detection', 'Segmentation', 'Total'], 
            [sizes['detection'], sizes['segmentation'], sizes['total']], 
            color=['blue', 'green', 'red'])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'model_size.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(['FPS'], [metrics['fps']], color='orange')
    plt.ylabel('Frames Per Second')
    plt.title('Model Throughput (FPS)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'fps.png'))
    plt.close()
    
    logger.info(f"Performance plots saved to {output_dir}")

def main():
    """
    Main execution function with improved training and evaluation pipeline
    """
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Output directory: {CURRENT_RUN_DIR}")
    
    det_model = train_detection_model()
    seg_model = train_segmentation_model()
    
    eval_metrics = evaluate_models(det_model, seg_model)
    
    perf_metrics = measure_inference_performance(det_model, seg_model)
    
    logger.info("\n======== Separate Models Performance ========")
    logger.info(f"Sequential Inference Time: {perf_metrics['sequential_inference_time']:.2f} ms")
    logger.info(f"Throughput: {perf_metrics['fps']:.2f} FPS")
    logger.info(f"Detection Model Size: {perf_metrics['model_sizes']['detection']:.2f} MB")
    logger.info(f"Segmentation Model Size: {perf_metrics['model_sizes']['segmentation']:.2f} MB")
    logger.info(f"Total Model Size: {perf_metrics['model_sizes']['total']:.2f} MB")
    logger.info(f"Combined Memory Usage: {perf_metrics['memory_usage']['combined']:.2f} MB")
    
    create_performance_plots(perf_metrics, CURRENT_RUN_DIR)
    
    if perf_metrics['test_images']:
        sample_img = perf_metrics['test_images'][0]
        output_img = os.path.join(CURRENT_RUN_DIR, "separate_inference_visualization.jpg")
        visualize_results(det_model, seg_model, sample_img, output_img)
    
    logger.info(f"Training and evaluation completed. Results saved to {CURRENT_RUN_DIR}")

if __name__ == "__main__":
    main()
