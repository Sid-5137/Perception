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

DATA_FRACTION = 1.0
EPOCHS = 50
PATIENCE = 10
SAVE_PERIOD = 5

def setup_directories():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "runs", current_time)
    
    paths = {
        'detection_weights': os.path.join(OUTPUT_DIR, "detection_only", "weights"),
        'segmentation_weights': os.path.join(OUTPUT_DIR, "segmentation_only", "weights"),
        'run_dir': run_dir,
        'eval_dir': os.path.join(run_dir, "evaluations")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def safe_serialize(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def train_detection_model(paths):
    logger.info("Training detection model...")
    
    try:
        model = YOLO('yolo11n.pt')
        results = model.train(
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
            val=True
        )

        metrics = {
            'map50': results.box.map50,
            'map': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'fitness': results.fitness,
            'epochs': results.epoch
        }
        
        with open(os.path.join(paths['run_dir'], 'detection_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=safe_serialize)
        
        model.export(format="onnx", imgsz=640, simplify=True)
        return model

    except Exception as e:
        logger.error(f"Detection training failed: {str(e)}")
        raise

def train_segmentation_model(paths):
    logger.info("Training segmentation model...")
    
    try:
        model = YOLO('yolo11n-seg.pt')
        results = model.train(
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
            val=True
        )

        metrics = {
            'seg_map50': results.seg.map50,
            'seg_map': results.seg.map,
            'miou': getattr(results.seg, 'miou', None),
            'box_map50': results.box.map50,
            'box_map': results.box.map,
            'precision': results.seg.mp,
            'recall': results.seg.mr,
            'fitness': results.fitness,
            'epochs': results.epoch
        }
        
        with open(os.path.join(paths['run_dir'], 'segmentation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=safe_serialize)
        
        model.export(format="onnx", imgsz=640, simplify=True)
        return model

    except Exception as e:
        logger.error(f"Segmentation training failed: {str(e)}")
        raise

def load_trained_model(model_type):
    weights_dir = os.path.join(OUTPUT_DIR, f"{model_type}_only", "weights")
    model_path = os.path.join(weights_dir, "best.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained {model_type} model found at {model_path}")
    
    return YOLO(model_path)

def evaluate_model(model, data_path, model_type):
    logger.info(f"Evaluating {model_type} model...")
    
    results = model.val(data=data_path)
    
    metrics = {
        'mAP50': results.box.map50 if model_type == 'detection' else results.seg.map50,
        'mAP': results.box.map if model_type == 'detection' else results.seg.map,
        'precision': results.box.mp if model_type == 'detection' else results.seg.mp,
        'recall': results.box.mr if model_type == 'detection' else results.seg.mr
    }
    
    if model_type == 'segmentation':
        metrics['mIoU'] = getattr(results.seg, 'miou', None)
    
    return metrics

def measure_inference_performance(det_model, seg_model, paths):
    logger.info("Measuring inference performance...")
    
    test_dir = os.path.join(os.path.dirname(BBOX_DATA_PATH), "test", "images")
    test_images = []
    
    if os.path.exists(test_dir):
        for img_file in os.listdir(test_dir)[:5]:
            img_path = os.path.join(test_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
                test_images.append((img_path, img_tensor))
    
    if not test_images:
        logger.warning("Using synthetic data for benchmarking")
        test_images = [("synthetic", torch.rand(3, 640, 640).to(DEVICE)) for _ in range(5)]

    times = []
    for img_path, img_tensor in test_images:
        start = time.time()
        det_model(img_tensor)
        seg_model(img_tensor)
        times.append(time.time() - start)
        logger.info(f"Processed {os.path.basename(img_path)} in {times[-1]*1000:.2f}ms")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    det_model(test_images[0][1])
    mem_after_det = process.memory_info().rss
    seg_model(test_images[0][1])
    mem_after_seg = process.memory_info().rss

    metrics = {
        'avg_inference_time': np.mean(times) * 1000,
        'fps': 1 / np.mean(times),
        'memory_usage': {
            'detection': (mem_after_det - mem_before) / (1024 ** 2),
            'segmentation': (mem_after_seg - mem_after_det) / (1024 ** 2)
        },
        'model_sizes': {
            'detection': os.path.getsize(os.path.join(OUTPUT_DIR, "detection_only", "weights", "best.pt")) / (1024 ** 2),
            'segmentation': os.path.getsize(os.path.join(OUTPUT_DIR, "segmentation_only", "weights", "best.pt")) / (1024 ** 2)
        }
    }
    
    with open(os.path.join(paths['eval_dir'], 'performance.json'), 'w') as f:
        json.dump(metrics, f, indent=4, default=safe_serialize)
    
    return metrics

def visualize_results(det_model, seg_model, img_path, output_dir):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0

        with torch.no_grad():
            start = time.time()
            det_results = det_model(img_tensor)
            det_time = time.time() - start
            
            start = time.time()
            seg_results = seg_model(img_tensor)
            seg_time = time.time() - start

        det_viz = det_results[0].plot()
        seg_viz = seg_results[0].plot()
        
        combined = np.hstack((det_viz, seg_viz))
        output_path = os.path.join(output_dir, "visualization.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        return output_path
    
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return None

def main():
    paths = setup_directories()
    
    try:
        det_model = train_detection_model(paths)
        seg_model = train_segmentation_model(paths)
        
        det_metrics = evaluate_model(det_model, BBOX_DATA_PATH, 'detection')
        seg_metrics = evaluate_model(seg_model, SEG_DATA_PATH, 'segmentation')
        
        with open(os.path.join(paths['eval_dir'], 'metrics.json'), 'w') as f:
            json.dump({
                'detection': det_metrics,
                'segmentation': seg_metrics
            }, f, indent=4, default=safe_serialize)
        
        perf_metrics = measure_inference_performance(det_model, seg_model, paths)
        
        sample_image = os.path.join(os.path.dirname(BBOX_DATA_PATH), "test", "images", "MP_SEL_SUR_000002.jpg")
        viz_path = visualize_results(det_model, seg_model, sample_image, paths['eval_dir'])
        
        logger.info("\n=== Final Results ===")
        logger.info(f"Detection mAP50: {det_metrics['mAP50']:.4f}")
        logger.info(f"Segmentation mIoU: {seg_metrics.get('mIoU', 'N/A')}")
        logger.info(f"Average Inference Time: {perf_metrics['avg_inference_time']:.2f}ms")
        logger.info(f"Results saved to: {paths['run_dir']}")
        
        if viz_path:
            logger.info(f"Visualization saved to: {viz_path}")

    except Exception as e:
        logger.error(f"Main workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
