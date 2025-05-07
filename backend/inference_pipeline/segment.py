import os
import logging
import torch
from collections import OrderedDict
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    Activationsd,
    AsDiscreted,
    Invertd,
    SaveImaged,
    Lambdad,
)
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet

from inference_pipeline.utils.dicom_dataset import DicomDataset
from inference_pipeline.utils.transforms import ExtractHeartRegiond, convert_multi_channel_to_single


def run_segmentation_inference(cfg):
    """Run inference using a trained model checkpoint"""

    # Extract config values
    checkpoint_path = cfg["segmentation_model_checkpoint_path"]
    dataset_root_dir = cfg["dataset_root_dir"]
    output_dir = os.path.join(cfg["output_dir"], "segmentation")
    dataset_split_json = cfg["dataset_json_path"]
    device = f"cuda:{cfg['device']}" if isinstance(cfg["device"], int) else "cpu"
    batch_size = 1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    # Load model using VNet directly
    model = VNet(in_channels=1, out_channels=7, spatial_dims=3)
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    new_state_dict = OrderedDict(
        (key.replace("model.", ""), value) for key, value in state_dict.items()
    )
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Setup transforms
    inference_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Resized(keys=["image"], spatial_size=(256, 256, 144), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ExtractHeartRegiond(
                keys="image",
                bloodpool_model_path=cfg["bp_model_checkpoint_path"],
                device=device,
            ),
        ]
    )

    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",
                transform=inference_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            Lambdad(
                keys="pred",
                func=lambda x: convert_multi_channel_to_single(x, n_classes=7),
            ),
            SaveImaged(
                keys="pred",
                output_dir=output_dir,
                output_postfix="seg",
                resample=False,
                output_ext=".nii.gz",
                data_root_dir=dataset_root_dir,
                separate_folder=False,
            ),
        ]
    )

    # Setup dataset and dataloader
    test_ds = DicomDataset(
        dataset_root_dir=dataset_root_dir,
        section="testing",
        transform=inference_transforms,
        splits_json=dataset_split_json,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Run inference
    with torch.no_grad():
        for batch_data in test_loader:
            # Move input data to device
            inputs = batch_data["image"].to(device)

            # Run inference with sliding window
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=(256, 256, 144),
                sw_batch_size=1,
                predictor=model.forward,
                overlap=0.5,
            )

            # Prepare batch for post-processing
            batch_data["pred"] = outputs

            # Post-process and save results
            processed_outputs = [
                post_transforms(i) for i in decollate_batch(batch_data)
            ]

            logging.info(f"Processed and saved predictions for batch")

    logging.info(f"Inference completed. Results saved to: {output_dir}")
