import modal

app = modal.App("vr-heart-backend")

# CPU image for FastAPI
cpu_image = (
    modal.Image.debian_slim()
    .apt_install("libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install_from_requirements("../requirements.txt")
    .add_local_python_source("backend")
    .add_local_python_source("inference_pipeline")
)

# GPU image for model inference
gpu_image = cpu_image  # same image, just used with GPU when called

# --- FastAPI ASGI app (CPU only) ---
@app.function(image=cpu_image, timeout=900)
@modal.asgi_app()
def fastapi_app():
    from backend import app as fastapi_instance
    return fastapi_instance

# --- Inference runner (GPU only) ---
@app.function(image=gpu_image, gpu="A100", timeout=900)
def run_inference(zip_bytes: bytes) -> bytes:
    import tempfile
    import os, uuid, zipfile
    from backend import process_dicom_zip

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(zip_bytes)
    tmp.flush()

    result = process_dicom_zip(
        zip_path=tmp.name,
        work_dir=tempfile.gettempdir(),
        out_dir=tempfile.gettempdir()
    )

    result_zip_path = os.path.join(tempfile.gettempdir(), f"results_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(result_zip_path, "w") as zout:
        for path in result["output_files"]:
            zout.write(path, arcname=os.path.basename(path))

    return open(result_zip_path, "rb").read()
