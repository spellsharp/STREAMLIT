import os
import logging
import time
import torch
import modal
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra import initialize, compose

from preprocess import preprocess_dicoms, make_json
from bloodpool import run_bp_inference
from segment import run_segmentation_inference

def setup_device(cfg: dict):
    """Setup device for inference, cfg is a dict here"""
    dev = cfg.get("device", None)
    if isinstance(dev, int) and dev >= 0:
        return f"cuda:{dev}"
    return "cpu"

image = (
    modal.Image
      .from_registry("pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime")
      .pip_install_from_requirements("../../requirements.txt")
      .add_local_python_source("preprocess","bloodpool","segment","utils")
)

cfg0 = OmegaConf.load("configs/config.yaml")

dataset_mount = modal.Mount.from_local_dir(
    local_path=cfg0.input_dicom_path,
    remote_path=cfg0.input_dicom_path,
)
bp_model_mount = modal.Mount.from_local_dir(
    local_path=os.path.dirname(cfg0.bp_model_checkpoint_path),
    remote_path=os.path.dirname(cfg0.bp_model_checkpoint_path),
)
seg_model_mount = modal.Mount.from_local_dir(
    local_path=os.path.dirname(cfg0.segmentation_model_checkpoint_path),
    remote_path=os.path.dirname(cfg0.segmentation_model_checkpoint_path),
)
output_mount = modal.Mount.from_local_dir(
    local_path=cfg0.output_dir,
    remote_path=cfg0.dataset_root_dir,
)
config_mount = modal.Mount.from_local_dir(
    local_path="configs",
    remote_path="/root/configs",
)

app = modal.App(image=image)

@app.function(
    mounts=[
      dataset_mount,
      bp_model_mount,
      seg_model_mount,
      output_mount,
      config_mount,
    ],
    gpu="A100"
)
def run_pipeline(config: dict):
    logging.basicConfig(level=logging.INFO)    
    logging.info(f"CUDA available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}")
    total_start = time.time()

    # PREPROCESSING
    logging.info("🛠️ Preprocessing data...")
    processed = []
    patient_folders = [
        d for d in os.listdir(config["input_dicom_path"])
        if os.path.isdir(os.path.join(config["input_dicom_path"], d))
    ]
    for pf in tqdm(patient_folders, desc="Patients"):
        inp = os.path.join(config["input_dicom_path"], pf)
        out = os.path.join(config["dataset_root_dir"], pf)
        os.makedirs(out, exist_ok=True)
        p = preprocess_dicoms(inp, out)
        if p:
            processed.append(p)
    make_json(processed, config["dataset_json_path"])
    logging.info("🛠️ Preprocessing done")

    # BLOOD-POOL
    logging.info("🚀 Running Blood-Pool inference...")
    bp_start = time.time()
    run_bp_inference(config)
    bp_time = time.time() - bp_start
    logging.info(f"⏱️ Blood-Pool done in {bp_time:.1f}s")

    # SEGMENTATION
    logging.info("🚀 Running Segmentation inference...")
    seg_start = time.time()
    run_segmentation_inference(config)
    seg_time = time.time() - seg_start
    logging.info(f"⏱️ Segmentation done in {seg_time:.1f}s")

    total_time = time.time() - total_start
    logging.info(f"✅ Total pipeline time: {total_time:.1f}s")

# 5) Local entrypoint just composes and fires off the remote
@app.local_entrypoint()
def main():
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(config_name="config")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # schedule + wait
    run_pipeline.remote(cfg_dict).wait()