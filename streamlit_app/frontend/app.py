import os
import sys
import uuid
import zipfile
import logging
import shutil
import time
import streamlit as st
import torch

# === ensure we have a place to unpack into ===
os.makedirs("./files", exist_ok=True)

# Define model paths
BP_MODEL_PATH = "../../models/bloodpool_models/45wq9ozw.ckpt"
SEG_MODEL_PATH = "../../models/segmentation_models/mqt8dcfs.ckpt"

from omegaconf import OmegaConf
from streamlit_app.inference_pipeline.preprocess import preprocess_dicoms, make_json
from streamlit_app.inference_pipeline.bloodpool import run_bp_inference
from streamlit_app.inference_pipeline.segment import run_segmentation_inference
from 
# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_zip(src: str, dest: str) -> None:
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dest)


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
    dataset_json_path = os.path.abspath(os.path.join(session_dir, "data.json"))

    return {
        "input_dicom_path": input_path,
        "dataset_root_dir": session_dir,
        "output_dir": results_dir,
        "dataset_json_path": dataset_json_path,
        "bp_model_checkpoint_path": BP_MODEL_PATH,
        "segmentation_model_checkpoint_path": SEG_MODEL_PATH,
        "device": 0 if torch.cuda.is_available() else "cpu"
    }


def main():
    st.set_page_config(page_title="VR-Heart Inference Portal", layout="centered")
    st.markdown(
        """
        <style>
            .title { font-size:2.2em; font-weight:bold; text-align:center; color:#4CAF50; margin-bottom:20px; }
        </style>
        <div class="title">VR-Heart Inference Portal</div>
        """, unsafe_allow_html=True
    )

    uploaded_zip = st.file_uploader("Upload ZIP of DICOMs", type=["zip"])
    if not uploaded_zip:
        st.info("Please upload a ZIP file containing your DICOM folders.")
        return

    if st.button("Process Data"):
        # Save and extract ZIP
        zip_path = os.path.join(".", uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        tmp_dir = os.path.join("./files", f"tmp_{uuid.uuid4().hex}")
        os.makedirs(tmp_dir, exist_ok=True)
        extract_zip(zip_path, tmp_dir)

        # Determine DICOM input dir
        entries = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
        if entries:
            patient = entries[0]
            extract_root = os.path.join("./files", f"extracted_{patient}")
            if os.path.exists(extract_root): shutil.rmtree(extract_root)
            shutil.move(os.path.join(tmp_dir, patient), extract_root)
        else:
            extract_root = tmp_dir
        input_dir = resolve_input_dir(extract_root)

        # Prepare config
        cfg = make_config(input_dir, "./outputs")

        # Track overall time
        pipeline_start = time.time()

        # Step 1: Preprocessing
        step1_start = time.time()
        with st.spinner("Step 1/3: Preprocessing DICOMs..."):
            try:
                subfolders = [
                    d for d in os.listdir(cfg["input_dicom_path"])
                    if os.path.isdir(os.path.join(cfg["input_dicom_path"], d))
                ]
                if subfolders:
                    processed_item = preprocess_dicoms(
                        cfg["input_dicom_path"], cfg["dataset_root_dir"]
                    )
                    processed = [processed_item] if processed_item else []
                else:
                    out_dir = os.path.join(cfg["dataset_root_dir"], "patient")
                    os.makedirs(out_dir, exist_ok=True)
                    processed_item = preprocess_dicoms(
                        cfg["input_dicom_path"], out_dir
                    )
                    processed = [processed_item] if processed_item else []
                make_json(processed, cfg["dataset_json_path"])
                st.success("Preprocessing complete")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return
        step1_end = time.time()

        # Step 2: Blood-Pool Inference
        step2_start = time.time()
        with st.spinner("Step 2/3: Running Blood-Pool inference..."):
            try:
                run_bp_inference(cfg)
                st.success("Blood-Pool inference complete")
            except Exception as e:
                st.error(f"Blood-Pool inference failed: {e}")
                return
        step2_end = time.time()

        # Step 3: Segmentation Inference
        step3_start = time.time()
        with st.spinner("Step 3/3: Running Segmentation inference..."):
            try:
                run_segmentation_inference(cfg)
                st.success("Segmentation inference complete")
            except Exception as e:
                st.error(f"Segmentation inference failed: {e}")
                return
        step3_end = time.time()

        # Overall end
        pipeline_end = time.time()

        # Display timing summary
        st.info(
            f"⏱️ Timing: Preprocessing: {step1_end - step1_start:.1f}s | "
            f"Blood-Pool: {step2_end - step2_start:.1f}s | "
            f"Segmentation: {step3_end - step3_start:.1f}s | "
            f"Total: {pipeline_end - pipeline_start:.1f}s"
        )

        # Final: Download results
        st.success("Pipeline completed successfully!")
        for root, _, files in os.walk(cfg["output_dir"]):
            for fname in files:
                file_path = os.path.join(root, fname)
                st.download_button(
                    label=f"Download {fname}",
                    data=open(file_path, "rb").read(),
                    file_name=fname,
                )

        # Cleanup
        os.remove(zip_path)
        # shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
