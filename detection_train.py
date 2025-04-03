import os
import json
import torch
import logging
from datetime import datetime
from ultralytics import YOLO

BBOX_DATA_PATH = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/YOLO/YOLO_BBOX/data.yaml"
OUTPUT_DIR = "/home/sid/Desktop/Projects/Walking-Assistant/Perception/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FRACTION = 1.0
EPOCHS = 50
PATIENCE = 10
SAVE_PERIOD = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_detection_model():
    logger.info(f"Training detection model on {DATA_FRACTION * 100:.1f}% of data...")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "runs", "detection", current_time)
    os.makedirs(run_dir, exist_ok=True)

    weights_dir = os.path.join(OUTPUT_DIR, "detection_only", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    last_pt = os.path.join(weights_dir, "last.pt")

    try:
        if os.path.exists(last_pt):
            logger.info(f"Resuming training from checkpoint: {last_pt}")
            model = YOLO(last_pt)
        else:
            logger.info("Starting fresh training with YOLOv11 model...")
            model = YOLO("yolo11n.pt")

        results = model.train(
            data=BBOX_DATA_PATH,
            epochs=EPOCHS,
            imgsz=640,
            batch=8,
            device=DEVICE,
            fraction=DATA_FRACTION,
            name="detection_only",
            project=OUTPUT_DIR,
            resume=os.path.exists(last_pt),
            patience=PATIENCE,
            save_period=SAVE_PERIOD,
            cos_lr=True,
            amp=True,
            plots=True,
            val=True
        )

        # Debugging: Log available metrics
        logger.info("Available metrics in results_dict after training:")
        logger.info(results.results_dict)

        # Access detection-specific metrics with safety checks
        metrics = {
            "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),        # Bounding box mAP50
            "mAP": results.results_dict.get("metrics/mAP50-95(B)", 0),      # Bounding box mAP50-95
            "precision": results.results_dict.get("metrics/precision(B)", 0),  # Bounding box precision
            "recall": results.results_dict.get("metrics/recall(B)", 0),       # Bounding box recall
            "epochs": EPOCHS,
        }

        # Save metrics to JSON
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Export model in ONNX format
        onnx_path = os.path.join(weights_dir, "model.onnx")
        model.export(format="onnx", imgsz=640, simplify=True, output=onnx_path)
        logger.info(f"ONNX model saved to {onnx_path}")

    except Exception as e:
        logger.error(f"Detection training failed: {str(e)}")
        if os.path.exists(last_pt):
            logger.info(f"Last checkpoint available at: {last_pt}")
        raise

def load_model(model_type):
    weights_dir = os.path.join(OUTPUT_DIR, f"{model_type}_only", "weights")
    model_path = os.path.join(weights_dir, "best.pt")

    if not os.path.exists(model_path):
        available_files = "\n- ".join(os.listdir(weights_dir)) if os.path.exists(weights_dir) else "None"
        raise FileNotFoundError(
            f"No trained {model_type} model found at {model_path}\n"
            f"Available files in weights directory:\n- {available_files}"
        )

    logger.info(f"Loading {model_type} model from {model_path}")
    return YOLO(model_path).to(DEVICE)

def evaluate_detection_model():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(OUTPUT_DIR, "evaluations", current_time)
    os.makedirs(eval_dir, exist_ok=True)

    try:
        det_model = load_model("detection")

        logger.info("Evaluating detection model on test dataset...")
        det_results = det_model.val(data=BBOX_DATA_PATH, split="test")

        # Debugging: Print the full results_dict to inspect available keys
        logger.info("Available metrics in results_dict:")
        logger.info(det_results.results_dict)

        # Access detection-specific metrics with safety checks
        det_metrics = {
            "mAP50": det_results.results_dict.get("metrics/mAP50(B)", 0),        # Bounding box mAP50
            "mAP": det_results.results_dict.get("metrics/mAP50-95(B)", 0),      # Bounding box mAP50-95
            "precision": det_results.results_dict.get("metrics/precision(B)", 0),  # Bounding box precision
            "recall": det_results.results_dict.get("metrics/recall(B)", 0),       # Bounding box recall
        }

        # Save metrics to JSON
        with open(os.path.join(eval_dir, "detection_metrics.json"), "w") as f:
            json.dump(det_metrics, f, indent=4)

        # Log results
        logger.info("\n=== Detection Model Evaluation Results ===")
        logger.info(f"mAP50: {det_metrics['mAP50']:.4f}")
        logger.info(f"mAP: {det_metrics['mAP']:.4f}")
        logger.info(f"Precision: {det_metrics['precision']:.4f}")
        logger.info(f"Recall: {det_metrics['recall']:.4f}")
        logger.info(f"Results saved to: {eval_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # train_detection_model()
    evaluate_detection_model()