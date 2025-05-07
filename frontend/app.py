import streamlit as st
import requests

st.title("VR-Heart Inference")

uploaded_file = st.file_uploader("Upload your ZIP of DICOMs", type="zip")

if uploaded_file and st.button("Run Inference and Download Results"):
    with st.spinner("Running inference on GPU..."):
        try:
            response = requests.post(
                "https://spellsharp--vr-heart-backend-fastapi-app.modal.run/process/",
                files={"file": uploaded_file},
                timeout=900  # allow time for GPU inference
            )

            if response.status_code == 200:
                st.success("Inference complete! Click below to download.")
                st.download_button(
                    label="📦 Download Results",
                    data=response.content,
                    file_name="inference_results.zip",
                    mime="application/zip"
                )
            else:
                st.error(f"Failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Error during inference: {e}")
