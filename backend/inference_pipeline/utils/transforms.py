import os
import json
from monai.transforms import (
    Compose,
    MapTransform,
    LabelFilter,
    AsDiscrete,
    InvertibleTransform,
    Activations,
)
from monai.data.meta_tensor import MetaTensor
from monai.networks.nets import VNet
import logging
import torch
import traceback
from collections import OrderedDict
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import numpy as np


def convert_multi_channel_to_single(multi_channel_mask, n_classes=7):
    """Convert multi-channel mask to single channel"""

    # logging.debug(f"Input tensor shape: {multi_channel_mask.shape}")
    if n_classes > 2:
        single_channel_mask = torch.argmax(multi_channel_mask, dim=0)
    else:
        single_channel_mask = multi_channel_mask.squeeze(0)
    # logging.debug(f"Output tensor shape: {single_channel_mask.shape}")
    return single_channel_mask


class ProcessLabelsTransformd(MapTransform, InvertibleTransform):
    def __init__(
        self,
        keys,
        dataset_root_dir,
        training_method,
        training_class_names,
    ):
        super().__init__(keys)

        dataset_json_path = os.path.join(dataset_root_dir, "classnames.json")
        with open(dataset_json_path, "r") as f:
            class_names = json.load(f)
        self.total_num_classes = len(class_names)

        self.applied_labels = [
            class_names[k]
            for k in list(
                set(class_names.keys()).intersection(set(training_class_names))
            )
        ]

        if training_method == "sequential":
            k = training_class_names[0]  # there will only be one class
            self.applied_labels = [class_names[k]]
            logging.debug(f"🚀 Included labels: {self.applied_labels}")
            self.remove_background = True
            self.remove_myocardium = True
        else:
            excluded_classes = list(
                set(class_names.keys()).difference(set(training_class_names))
            )
            logging.debug(f"🚀 Excluded classes: {excluded_classes}")
            self.remove_background = "background" not in training_class_names
            self.remove_myocardium = "myocardium" not in training_class_names

        self.training_method = training_method
        self.training_class_names = training_class_names
        self.myocardium_index = class_names["myocardium"]
        self.label_filter = LabelFilter(applied_labels=self.applied_labels)
        self.as_discrete = AsDiscrete(to_onehot=self.total_num_classes)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            if key not in d:
                continue

            extra_info = {
                "total_num_classes": self.total_num_classes,
                "applied_labels": self.applied_labels,
                "remove_background": self.remove_background,
                "remove_myocardium": self.remove_myocardium,
                "myocardium_index": self.myocardium_index,
                "training_method": self.training_method,
                "training_class_names": self.training_class_names,
            }

            self.push_transform(d[key], extra_info=extra_info)

            applied_operations = d[key].applied_operations

            original = d[key]
            meta = original.meta if isinstance(original, MetaTensor) else None

            if self.training_method == "blood_pool":
                d[key] = torch.where(d[key] > 0, 1, 0)
                self.total_num_classes = 1
                return d

            # Filter labels using LabelFilter
            if self.applied_labels:
                d[key] = self.label_filter(d[key])

            # One-hot encoding
            one_hot = self.as_discrete(d[key])

            # For sequential training, only keep the channel corresponding to the class
            if len(self.applied_labels) == 1:
                result = one_hot[self.applied_labels[0] : self.applied_labels[0] + 1]
            else:
                # Remove specified channels
                channels_to_keep = []
                for i in range(self.total_num_classes):
                    if i == 0 and self.remove_background:
                        continue
                    if i == self.myocardium_index and self.remove_myocardium:
                        continue
                    channels_to_keep.append(i)
                result = one_hot[channels_to_keep]

            # Restore metatensor properties
            if meta is not None:
                result = MetaTensor(result, meta=meta)
                result.applied_operations = applied_operations
            d[key] = result

        return d

    def inverse(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            # Get the transform data
            transform = self.pop_transform(d[key])
            # Apply inverse transform
            d[key] = self.inverse_transform(d[key], transform)
        return d

    def inverse_transform(self, data: MetaTensor, transform) -> MetaTensor:

        n_classes = transform["extra_info"]["total_num_classes"]
        # Ensure data is in the correct shape before converting
        # if len(data.shape) < 3:  # If data is 2D, add channel dimension
        #     data = data.unsqueeze(0)
        # Convert multi-channel to single channel
        output_mask = convert_multi_channel_to_single(data, n_classes)

        # Preserve metatensor properties
        if isinstance(data, MetaTensor):
            output_mask = MetaTensor(output_mask, meta=data.meta)
            # Preserve the applied operations except for the current transform
            if hasattr(data, "applied_operations"):
                output_mask.applied_operations = data.applied_operations

        return output_mask


class ExtractHeartRegiond(MapTransform):
    def __init__(
        self,
        keys,
        bloodpool_model_path,
        use_labels=False,
        padding_method="dilation",
        sw_inference=False,
        device=None,
    ):
        super().__init__(keys)
        self.sw_inference = sw_inference
        self.padding_method = padding_method
        self.bloodpool_model_path = bloodpool_model_path
        self.use_labels = use_labels
        self.device = "cpu"

        if not self.use_labels:
            self.bloodpool_model = self._load_model()

        self.post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

    def _load_model(self):
        model = VNet(in_channels=1, out_channels=1, spatial_dims=3)
        state_dict = torch.load(self.bloodpool_model_path, map_location=self.device)[
            "state_dict"
        ]
        new_state_dict = OrderedDict(
            (key.replace("model.", ""), value) for key, value in state_dict.items()
        )
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def __call__(self, data):
        for key in self.keys:
            if key not in data:
                continue
            original_image = data[key]
            # Always use self.device to ensure consistency
            device = self.device

            if isinstance(original_image, torch.Tensor):
                # Ensure we have a single channel and batch dimension
                if len(original_image.shape) == 3:
                    original_image_tensor = (
                        original_image.clone()
                        .detach()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )
                elif len(original_image.shape) == 4:  # Has channel dimension
                    original_image_tensor = (
                        original_image.clone().detach().unsqueeze(0).float().to(device)
                    )
                elif len(original_image.shape) == 5:  # Already has batch and channel
                    original_image_tensor = (
                        original_image.clone().detach().float().to(device)
                    )
            else:
                original_image_tensor = (
                    torch.from_numpy(original_image)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )

            # Ensure the image tensor is on the correct device
            original_image_tensor = original_image_tensor.to(device)

            if not self.use_labels:
                with torch.inference_mode():
                    prediction = self.bloodpool_model(original_image_tensor)
                    prediction = self.post_trans(prediction)
            else:
                prediction = convert_multi_channel_to_single(
                    data["label"]
                )  # Use labels instead of prediction for training for more accurate training
                logging.debug(
                    f"Using labels for heart region extraction. Labels shape: {prediction.shape}"
                )
            prediction = prediction.squeeze().detach().cpu()
            heart_mask = prediction > 0

            if self.padding_method == "dilation":
                heart_mask_np = heart_mask.numpy()
                padded_heart_mask_np = binary_dilation(heart_mask_np, iterations=10)
                padded_heart_mask = torch.from_numpy(padded_heart_mask_np).to(device)
            elif self.padding_method == "radius":
                heart_mask_np = heart_mask.numpy()
                if np.sum(heart_mask_np) > 0:
                    indices = np.array(np.nonzero(heart_mask_np))
                    center = np.mean(indices, axis=1).astype(int)

                    radius = 100  # Adjust this value based on your needs
                    x, y, z = np.ogrid[
                        : heart_mask_np.shape[0],
                        : heart_mask_np.shape[1],
                        : heart_mask_np.shape[2],
                    ]
                    dist_from_center = (
                        (x - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (z - center[2]) ** 2
                    )
                    sphere_mask = dist_from_center <= radius**2
                    padded_heart_mask_np = np.logical_or(heart_mask_np, sphere_mask)
                else:
                    padded_heart_mask_np = binary_dilation(heart_mask_np, iterations=10)

                padded_heart_mask = torch.from_numpy(padded_heart_mask_np).to(device)
            elif self.padding_method is None:
                padded_heart_mask = heart_mask.to(device)

            if len(padded_heart_mask.shape) < len(original_image_tensor.shape):
                for _ in range(
                    len(original_image_tensor.shape) - len(padded_heart_mask.shape)
                ):
                    padded_heart_mask = padded_heart_mask.unsqueeze(0)

            heart_region = torch.where(
                padded_heart_mask,
                original_image_tensor,
                torch.zeros_like(original_image_tensor),
            )

            if len(original_image.shape) < len(heart_region.shape):
                heart_region = heart_region.squeeze(0)

            heart_region = heart_region.cpu()
            data[key] = heart_region  # overwrite image key

        return data
