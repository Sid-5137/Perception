import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from ultralytics import YOLO

SEG_DATA_PATH = "/media/sid/Sid-HDD/Datasets/Vision/Sideguide/YOLO/YOLO_Seg/data.yaml"
OUTPUT_DIR = "/home/sid/Desktop/Projects/Walking-Assistant/Perception/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FRACTION = 1.0
EPOCHS = 50
PATIENCE = 10
SAVE_PERIOD = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_segmentation_model():
    logger.info(f"Training segmentation model on {DATA_FRACTION * 100:.1f}% of data...")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "runs", "segmentation", current_time)
    os.makedirs(run_dir, exist_ok=True)

    weights_dir = os.path.join(OUTPUT_DIR, "segmentation_only", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    last_pt = os.path.join(weights_dir, "last.pt")

    try:
        if os.path.exists(last_pt):
            logger.info(f"Resuming training from checkpoint: {last_pt}")
            model = YOLO(last_pt)
        else:
            logger.info("Starting fresh training with YOLOv11 segmentation model...")
            model = YOLO("yolo11n-seg.pt")

        results = model.train(
            data=SEG_DATA_PATH,
            epochs=EPOCHS,
            imgsz=640,
            batch=8,
            device=DEVICE,
            fraction=DATA_FRACTION,
            name="segmentation_only",
            project=OUTPUT_DIR,
            resume=os.path.exists(last_pt),
            patience=PATIENCE,
            save_period=SAVE_PERIOD,
            cos_lr=True,
            amp=True,
            plots=True,
            val=True
        )

        logger.info("Available metrics in results_dict after training:")
        logger.info(results.results_dict)

        metrics = {
            "seg_mAP50": results.results_dict.get("metrics/mAP50(M)", 0),
            "seg_mAP": results.results_dict.get("metrics/mAP50-95(M)", 0),
            "seg_precision": results.results_dict.get("metrics/precision(M)", 0),
            "seg_recall": results.results_dict.get("metrics/recall(M)", 0),
            "epochs": EPOCHS,
        }

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        onnx_path = os.path.join(weights_dir, "model.onnx")
        model.export(format="onnx", imgsz=640, simplify=True, output=onnx_path)
        logger.info(f"ONNX model saved to {onnx_path}")

    except Exception as e:
        logger.error(f"Segmentation training failed: {str(e)}")
        if os.path.exists(last_pt):
            logger.info(f"Last checkpoint available at: {last_pt}")
        raise

def calculate_miou(seg_results):
    try:
        # Check if confusion matrix is available
        if hasattr(seg_results, "confusion_matrix") and seg_results.confusion_matrix is not None:
            confusion_matrix = seg_results.confusion_matrix.matrix  # Access confusion matrix
            logger.info(f"Confusion matrix shape: {confusion_matrix.shape}")
        else:
            logger.warning("No confusion matrix available, returning mIoU = 0.0")
            return 0.0

        intersection = np.diag(confusion_matrix)
        union = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection)

        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        valid_ious = iou[confusion_matrix.sum(axis=1) > 0]
        miou = np.nanmean(valid_ious) if len(valid_ious) > 0 else 0.0

        logger.info(f"Calculated mIoU: {miou:.4f}")
        return miou

    except Exception as e:
        logger.error(f"mIoU calculation failed: {str(e)}")
        return 0.0

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

def evaluate_segmentation_model():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(OUTPUT_DIR, "evaluations", current_time)
    os.makedirs(eval_dir, exist_ok=True)

    try:
        seg_model = load_model("segmentation")
        logger.info("Evaluating segmentation model on test dataset...")
        seg_results = seg_model.val(data=SEG_DATA_PATH, split="test")

        logger.info("Available metrics in results_dict:")
        logger.info(seg_results.results_dict)

        seg_metrics = {
            "mAP50": seg_results.results_dict.get("metrics/mAP50(M)", 0),
            "mAP": seg_results.results_dict.get("metrics/mAP50-95(M)", 0),
            "mIoU": calculate_miou(seg_results),
            "precision": seg_results.results_dict.get("metrics/precision(M)", 0),
            "recall": seg_results.results_dict.get("metrics/recall(M)", 0),
        }

        with open(os.path.join(eval_dir, "segmentation_metrics.json"), "w") as f:
            json.dump(seg_metrics, f, indent=4)

        logger.info("\n=== Segmentation Model Evaluation Results ===")
        logger.info(f"mAP50: {seg_metrics['mAP50']:.4f}")
        logger.info(f"mAP: {seg_metrics['mAP']:.4f}")
        logger.info(f"mIoU: {seg_metrics['mIoU']:.4f}")
        logger.info(f"Precision: {seg_metrics['precision']:.4f}")
        logger.info(f"Recall: {seg_metrics['recall']:.4f}")
        logger.info(f"Results saved to: {eval_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # train_segmentation_model()
    evaluate_segmentation_model()