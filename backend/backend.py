import os
import uuid
import zipfile
import shutil
import tempfile
import time
import torch
import traceback
import logging
import modal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from inference_pipeline.preprocess import preprocess_dicoms, make_json
from inference_pipeline.bloodpool import run_bp_inference
from inference_pipeline.segment import run_segmentation_inference
from modal_app import run_inference

# -------------------------------------------------
# Turn on debug logging and FastAPI’s debug mode
# -------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI(debug=True)
# -------------------------------------------------

app = FastAPI()

_model_paths: dict[str, str] = {}

def get_model_path(model_type: str) -> str:
    if model_type not in _model_paths:
        repo_id = "spellsharp/vr-heart-model"
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"models/{model_type}_models/baseline.ckpt"
        )
        _model_paths[model_type] = ckpt_path
    return _model_paths[model_type]

def resolve_input_dir(base_dir: str) -> str:
    for root, _, files in os.walk(base_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            return root
    return base_dir

def make_config(input_path: str, output_base_dir: str) -> dict:
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(output_base_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    results_dir = os.path.join(session_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    dataset_json_path = os.path.join(session_dir, "data.json")
    return {
        "input_dicom_path": input_path,
        "dataset_root_dir": session_dir,
        "output_dir": results_dir,
        "dataset_json_path": dataset_json_path,
        "bp_model_checkpoint_path": get_model_path("bloodpool"),
        "segmentation_model_checkpoint_path": get_model_path("segmentation"),
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }

def extract_zip(src: str, dest: str) -> None:
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dest)

def cleanup(paths: list[str]) -> None:
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)

def process_dicom_zip(zip_path: str, work_dir: str, out_dir: str) -> dict:
    # 1) Unpack
    tmp_dir = os.path.join(work_dir, f"tmp_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    extract_zip(zip_path, tmp_dir)

    # 2) Locate DICOM folder
    subdirs = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
    if subdirs:
        # Build a fresh extraction folder name so no collisions can occur
        folder_base = subdirs[0]
        unique_name = f"extracted_{folder_base}_{uuid.uuid4().hex}"
        extract_root = os.path.join(work_dir, unique_name)
        shutil.move(os.path.join(tmp_dir, folder_base), extract_root)
    else:
        extract_root = tmp_dir


    input_dir = resolve_input_dir(extract_root)
    cfg = make_config(input_dir, out_dir)

    # 3) Run pipeline
    timings: dict[str, float] = {}
    t0 = time.time()

    t1 = time.time()
    subfolders = os.listdir(cfg["input_dicom_path"])
    target_dir = cfg["dataset_root_dir"] if subfolders else os.path.join(cfg["dataset_root_dir"], "patient")
    os.makedirs(target_dir, exist_ok=True)
    processed = preprocess_dicoms(cfg["input_dicom_path"], target_dir)
    make_json([processed] if processed else [], cfg["dataset_json_path"])
    timings["preprocessing"] = time.time() - t1

    t2 = time.time()
    run_bp_inference(cfg)
    timings["bloodpool"] = time.time() - t2

    t3 = time.time()
    run_segmentation_inference(cfg)
    timings["segmentation"] = time.time() - t3

    timings["total"] = time.time() - t0

    # 4) Collect outputs
    output_files: list[str] = []
    for root, _, files in os.walk(cfg["output_dir"]):
        for fname in files:
            output_files.append(os.path.join(root, fname))

    # 5) Cleanup temp
    cleanup([tmp_dir])

    return {"cfg": cfg, "timings": timings, "output_files": output_files}

modal_app = modal.App.lookup("vr-heart-backend")


@app.post("/process/")
async def process_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result_bytes = run_inference.remote(content)
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.write(result_bytes)
        tmp.flush()
        return FileResponse(tmp.name, filename="results.zip", media_type="application/zip")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
