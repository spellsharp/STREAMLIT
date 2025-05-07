import os
import shutil
import pydicom
import logging
from pydicom.dataelem import DataElement
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def make_json(dicom_paths, output_json_path, label_paths=None):
    """
    Create a JSON file with testing data entries containing image and label paths.

    Args:
        dicom_paths (list): List of paths to DICOM directories
        output_json_path (str): Path where the JSON file will be saved

    """
    # Create entries with both image and label paths if provided
    json_entries = []
    for dicom_path in dicom_paths:
        json_entry = {"image": dicom_path}
        json_entries.append(json_entry)

    # Create the final JSON structure with only testing section
    json_data = {"testing": json_entries}

    # Write to file
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)


def convert_value_to_vr(value, vr):
    """Convert the input value to the appropriate type based on the VR."""
    if isinstance(value, list):
        if vr in ["DS"]:
            return "\\".join(str(float(v)) for v in value)
        elif vr in ["IS", "SS", "US"]:
            return "\\".join(str(int(v)) for v in value)
        else:
            return "\\".join(str(v) for v in value)
    else:
        if vr in ["IS", "SS", "US"]:
            return int(value)
        elif vr in ["FL", "FD"]:
            return float(value)
        elif vr in ["DS"]:
            return str(float(value))
        else:
            return str(value)


def find_dicom_directory(patient_folder):
    """
    Find the DICOM directory within a patient folder.
    If no dedicated DICOM directory is found, check for .dcm files
    in the patient folder or any subdirectory.

    Args:
        patient_folder (str): Path to patient folder

    Returns:
        str: Path to DICOM directory or None if not found
    """
    # First check if the input directory name contains "dicom"
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        if os.path.isdir(item_path) and "dicom" in item.lower():
            return item_path

    # Next, check if there are .dcm files directly in the input directory
    if any(file.lower().endswith('.dcm') for file in os.listdir(patient_folder) 
           if os.path.isfile(os.path.join(patient_folder, file))):
        return patient_folder
    
    # Finally, look recursively for any directory containing .dcm files
    for root, _, files in os.walk(patient_folder):
        if any(file.lower().endswith('.dcm') for file in files):
            return root
    
    # No DICOM files found anywhere
    return None


def copy_dicom_directory(source_dir, output_dir):
    """
    Copy DICOM files to a new 'shifted_dicom' directory.

    Args:
        source_dir (str): Path to source DICOM directory
        output_dir (str): Path to patient output directory

    Returns:
        str: Path to the new shifted_dicom directory
    """
    target_dir = os.path.join(output_dir, "shifted_dicom")

    # Remove existing directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Create the target directory
    os.makedirs(target_dir)

    # Copy only DICOM files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".dcm"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                shutil.copy2(source_file, target_file)

    # logging.info(f"Copied DICOM files to: {target_dir}")
    return target_dir


def update_dicom_origin(dicom_path, new_origin):
    """
    Update the image position (patient) DICOM tag with new origin coordinates
    while preserving slice spacing.

    Args:
        dicom_path (str): Path to DICOM directory
        new_origin (list): New origin coordinates [x, y, z]
    """
    # First, get all DICOM files and sort them by position
    dicom_files = []
    for root, _, files in os.walk(dicom_path):
        for file in files:
            if file.endswith(".dcm"):
                filepath = os.path.join(root, file)
                ds = pydicom.dcmread(filepath)
                position = (
                    ds.ImagePositionPatient[2]
                    if hasattr(ds, "ImagePositionPatient")
                    else 0
                )
                dicom_files.append((position, filepath))

    # Sort files by z-position
    dicom_files.sort(key=lambda x: x[0])

    # Calculate slice thickness
    if len(dicom_files) > 1:
        first_ds = pydicom.dcmread(dicom_files[0][1])
        second_ds = pydicom.dcmread(dicom_files[1][1])
        slice_thickness = abs(
            second_ds.ImagePositionPatient[2] - first_ds.ImagePositionPatient[2]
        )
        # logging.info(f"Calculated slice thickness: {slice_thickness}")
    else:
        slice_thickness = 1.0
        logging.warning(
            "Could not calculate slice thickness, using default value of 1.0"
        )

    # Update each file while preserving z-spacing
    for idx, (_, filepath) in enumerate(dicom_files):
        ds = pydicom.dcmread(filepath)

        # Log original values
        # logging.info(
        #     f"Before modification - Pixel Spacing: {ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else 'Not set'}, "
        #     f"Original Position: {ds.ImagePositionPatient if hasattr(ds, 'ImagePositionPatient') else 'Not set'}"
        # )

        # Create new position: keep x,y from new_origin but calculate appropriate z
        new_position = list(new_origin)
        new_position[2] = new_origin[2] + (idx * slice_thickness)

        # Update position
        ds.ImagePositionPatient = new_position

        # Set slice thickness if it's missing
        if not hasattr(ds, "SliceThickness") or ds.SliceThickness is None:
            ds.SliceThickness = slice_thickness

        # Log new values
        # logging.info(
        #     f"After modification - Pixel Spacing: {ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else 'Not set'}, "
        #     f"New Position: {ds.ImagePositionPatient}, Slice Thickness: {ds.SliceThickness}"
        # )

        ds.save_as(filepath)


def update_dicom_tag(dicom_dir, tag, value):
    """
    Update DICOM tags in all files in the directory while preserving spacing.

    Args:
        dicom_dir (str): Path to DICOM directory
        tag (tuple): DICOM tag to update (group, element)
        value: New value for the tag
    """
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]

    # Sort files by their z-position if we're dealing with position tags
    if tag == (0x0020, 0x0032):  # ImagePositionPatient tag
        file_positions = []
        for dicom_file in dicom_files:
            file_path = os.path.join(dicom_dir, dicom_file)
            ds = pydicom.dcmread(file_path)
            if hasattr(ds, "ImagePositionPatient"):
                z_pos = ds.ImagePositionPatient[2]
                file_positions.append((z_pos, dicom_file))
        file_positions.sort(key=lambda x: x[0])
        dicom_files = [f[1] for f in file_positions]

        # Calculate z-spacing between slices
        if len(file_positions) > 1:
            z_spacing = file_positions[1][0] - file_positions[0][0]
            # logging.info(f"Original z-spacing between slices: {z_spacing}")

    # Process each file
    for idx, dicom_file in enumerate(dicom_files):
        file_path = os.path.join(dicom_dir, dicom_file)
        ds = pydicom.dcmread(file_path)

        # If we're updating position, maintain relative z-spacing
        if tag == (0x0020, 0x0032):
            modified_value = list(value)
            if len(file_positions) > 1:
                modified_value[2] = value[2] + (idx * z_spacing)
            vr = ds[tag].VR if tag in ds else "DS"
            converted_value = convert_value_to_vr(modified_value, vr)
        else:
            vr = ds[tag].VR if tag in ds else "LO"
            converted_value = convert_value_to_vr(value, vr)

        # Update the tag
        data_element = DataElement(tag, vr, converted_value)
        ds[tag] = data_element
        ds.save_as(file_path)


def preprocess_dicoms(input_path, output_dir, new_origin=None):
    """
    Main function to preprocess DICOM files.

    Args:
        input_path (str): Path to patient folder containing DICOM directory
        output_dir (str): Path to output directory for this patient
        new_origin (list, optional): New origin coordinates [x, y, z]. Defaults to [0, 0, 0]

    Returns:
        str: Path to the processed DICOM directory
    """
    if new_origin is None:
        new_origin = [0, 0, 0]

    try:
        # Find the DICOM directory
        dicom_dir = find_dicom_directory(input_path)
        if not dicom_dir:
            raise ValueError(f"No DICOM directory found in {input_path}")

        # Create a copy of the DICOM directory
        processed_dicom_path = copy_dicom_directory(dicom_dir, output_dir)

        # Update the origin
        update_dicom_origin(processed_dicom_path, new_origin)

        logging.info(f"Processed DICOM directory: {processed_dicom_path}")
        return processed_dicom_path

    except Exception as e:
        logging.error(f"Error preprocessing DICOMs: {str(e)}")
        raise
