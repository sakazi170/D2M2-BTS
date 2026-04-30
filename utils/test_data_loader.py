import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from scipy import ndimage


class BraTSDataset(Dataset):

    def __init__(self, data_root, train_dataset, labels=False, smart_crop=False, custom_crops_file=None):

        self.data_root = data_root
        self.train_dataset = train_dataset
        self.labels = labels
        self.smart_crop = smart_crop
        # Get all subject folders
        self.subject_paths = natsorted(glob.glob(os.path.join(data_root, '*/')))

        # Load custom crops if file is provided
        self.custom_crops = {}
        if custom_crops_file and os.path.exists(custom_crops_file):
            self.load_custom_crops(custom_crops_file)

    def load_custom_crops(self, file_path):

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name = parts[0]
                    crop_coords = parts[1]

                    # Parse the crop coordinates (x_start, y_start, z_start)
                    coords = list(map(int, crop_coords.split(",")))
                    if len(coords) == 3:
                        self.custom_crops[image_name] = coords

        print(f"Loaded {len(self.custom_crops)} custom crop coordinates")

    def __len__(self):
        return len(self.subject_paths)

    def estimate_tumor_region(self, t1, t1ce, t2, flair):
        # Create a combined abnormality map
        abnormality_map = np.zeros_like(t1)

        # Process each modality
        for idx, modality in enumerate([flair, t1, t1ce, t2]):
            # Z-score normalize
            mean = np.mean(modality)
            std = np.std(modality)
            norm_modality = (modality - mean) / (std + 1e-8)

            # Different thresholds for different modalities
            if idx == 0:  # FLAIR - sensitive to edema
                threshold = 2.0
                abnormality_map += (norm_modality > threshold).astype(float) * 1.5
            elif idx == 2:  # t1ce - sensitive to enhancing tumor
                threshold = 2.5
                abnormality_map += (norm_modality > threshold).astype(float) * 2.0
            else:  # t1 and t2
                threshold = 2.0
                abnormality_map += (norm_modality > threshold).astype(float)

        # Apply smoothing to reduce noise
        abnormality_map = ndimage.gaussian_filter(abnormality_map, sigma=1)

        # Find the largest connected component
        labeled_array, num_features = ndimage.label(abnormality_map > 1.5)
        if num_features > 0:
            sizes = ndimage.sum(abnormality_map > 1.5, labeled_array, range(1, num_features + 1))
            max_label = np.argmax(sizes) + 1
            largest_component = labeled_array == max_label

            # Get center of the largest component
            coords = np.where(largest_component)
            if len(coords[0]) > 0:
                center_x = int(np.mean(coords[0]))
                center_y = int(np.mean(coords[1]))
                center_z = int(np.mean(coords[2]))
                return center_x, center_y, center_z

        # Fallback to fixed crop coordinates if no clear abnormality is found
        return 120, 120, 77  # Center of typical fixed crop

    def crop_image(self, image, mask=None, tumor_center=None, filename=None):

        # First check if we have custom coordinates for this file
        if filename is not None and filename in self.custom_crops and not self.labels:
            x_start, y_start, z_start = self.custom_crops[filename]
            size = 128
            # Ensure we don't go out of bounds
            x_start = min(max(0, x_start), image.shape[0] - size)
            y_start = min(max(0, y_start), image.shape[1] - size)
            z_start = min(max(0, z_start), image.shape[2] - size)

            cropped = image[x_start:x_start + size,
                      y_start:y_start + size,
                      z_start:z_start + size]
            return cropped, (x_start, y_start, z_start)

        # Continue with existing logic for other cases
        if mask is not None and self.labels:
            # For labeled data, center crop around tumor
            tumor_coords = np.where(mask > 0)
            if len(tumor_coords[0]) == 0:  # If no tumor present
                return image[56:184, 56:184, 13:141], (56, 56, 13)

            # Calculate center of tumor
            center_x = int(np.mean(tumor_coords[0]))
            center_y = int(np.mean(tumor_coords[1]))
            center_z = int(np.mean(tumor_coords[2]))

        elif tumor_center is not None and self.smart_crop and not self.labels:
            # Use provided tumor center for smart cropping
            center_x, center_y, center_z = tumor_center

        else:
            # Use fixed crop
            return image[56:184, 56:184, 13:141], (56, 56, 13)

        # Calculate crop size (128x128x128)
        size = 128
        x_start = max(0, center_x - size // 2)
        y_start = max(0, center_y - size // 2)
        z_start = max(0, center_z - size // 2)

        # Adjust if too close to edges
        x_start = min(x_start, image.shape[0] - size)
        y_start = min(y_start, image.shape[1] - size)
        z_start = min(z_start, image.shape[2] - size)

        cropped = image[x_start:x_start + size,
                  y_start:y_start + size,
                  z_start:z_start + size]

        return cropped, (x_start, y_start, z_start)

    def z_score_normalization(self, image):
        # normalize only non-zero voxels, same as training
        mask = image > 0
        if mask.sum() == 0:
            return image
        mean = image[mask].mean()
        std = image[mask].std()
        out = image.copy()
        out[mask] = (image[mask] - mean) / (std + 1e-8)
        return out

    def get_file_patterns(self):
        """Get file patterns based on dataset type"""
        if self.train_dataset == 'brats2019':
            return {
                't1': '*t1.nii.gz',
                't1ce': '*t1ce.nii.gz',
                't2': '*t2.nii.gz',
                'flair': '*flair.nii.gz',
                'seg': '*seg.nii.gz'
            }
        elif self.train_dataset == 'brats2020':
            return {
                't1': '*t1.nii.gz',
                't1ce': '*t1ce.nii.gz',
                't2': '*t2.nii.gz',
                'flair': '*flair.nii.gz',
                'seg': '*seg.nii.gz'
            }
        elif self.train_dataset == 'brats2021':
            return {
                't1': '*-t1n.nii.gz',
                't1ce': '*-t1c.nii.gz',
                't2': '*-t2w.nii.gz',
                'flair': '*-t2f.nii.gz',
                'seg': '*-seg.nii.gz'
            }
        elif self.train_dataset == 'brats2023':
            return {
                't1': '*-t1n.nii.gz',
                't1ce': '*-t1c.nii.gz',
                't2': '*-t2w.nii.gz',
                'flair': '*-t2f.nii.gz',
                'seg': '*-seg.nii.gz'
            }
        else:
            raise ValueError(f'Unsupported dataset: {self.train_dataset}')

    def __getitem__(self, idx):
        subject_path = self.subject_paths[idx]
        patterns = self.get_file_patterns()

        if self.labels:
            # Get all modality files for the subject
            t1_file = glob.glob(os.path.join(subject_path, patterns['t1']))[0]
            t1ce_file = glob.glob(os.path.join(subject_path, patterns['t1ce']))[0]
            t2_file = glob.glob(os.path.join(subject_path, patterns['t2']))[0]
            flair_file = glob.glob(os.path.join(subject_path, patterns['flair']))[0]
            seg_file = glob.glob(os.path.join(subject_path, patterns['seg']))[0]

            # Load mask first
            mask = nib.load(seg_file).get_fdata().astype(np.uint8)

            # Load and preprocess each modality
            t1 = nib.load(t1_file).get_fdata()
            t1, crop_coords = self.crop_image(t1, mask)
            t1 = self.z_score_normalization(t1)

            t1ce = nib.load(t1ce_file).get_fdata()
            t1ce = self.crop_image(t1ce, mask)[0]  # Only take image, not coords
            t1ce = self.z_score_normalization(t1ce)

            t2 = nib.load(t2_file).get_fdata()
            t2 = self.crop_image(t2, mask)[0]
            t2 = self.z_score_normalization(t2)

            flair = nib.load(flair_file).get_fdata()
            flair = self.crop_image(flair, mask)[0]
            flair = self.z_score_normalization(flair)

            # Crop mask using same coordinates
            mask = self.crop_image(mask, mask)[0]
            # Only remap labels for BraTS 2020
            if self.train_dataset in ['brats2019', 'brats2020']:
                mask[mask == 4] = 3  # Reassign mask values 4 to 3

            # Convert to tensors
            t1 = torch.tensor(t1).float().unsqueeze(0)
            t1ce = torch.tensor(t1ce).float().unsqueeze(0)
            t2 = torch.tensor(t2).float().unsqueeze(0)
            flair = torch.tensor(flair).float().unsqueeze(0)
            mask = torch.tensor(mask).long()

            # Store original filename for saving predictions
            filename = os.path.basename(os.path.dirname(t1_file))

            return t1, t1ce, t2, flair, mask, filename, crop_coords
        else:
            # Get all modality files for the subject
            t1_file = glob.glob(os.path.join(subject_path, patterns['t1']))[0]
            t1ce_file = glob.glob(os.path.join(subject_path, patterns['t1ce']))[0]
            t2_file = glob.glob(os.path.join(subject_path, patterns['t2']))[0]
            flair_file = glob.glob(os.path.join(subject_path, patterns['flair']))[0]

            # Get filename for custom crop lookup - check both with and without extension
            base_filename = os.path.basename(os.path.dirname(t1_file))
            full_filename = f"{base_filename}.nii.gz"

            # Determine which filename to use for custom crop lookup
            lookup_filename = None
            if full_filename in self.custom_crops:
                lookup_filename = full_filename
            elif base_filename in self.custom_crops:
                lookup_filename = base_filename

            # Load all modalities
            t1_data = nib.load(t1_file).get_fdata()
            t1ce_data = nib.load(t1ce_file).get_fdata()
            t2_data = nib.load(t2_file).get_fdata()
            flair_data = nib.load(flair_file).get_fdata()

            # First priority: use custom crop if available
            if lookup_filename is not None:
                print(f"\nCustom crop for {lookup_filename}")
                t1, crop_coords = self.crop_image(t1_data, filename=lookup_filename)
                t1ce = self.crop_image(t1ce_data, filename=lookup_filename)[0]
                t2 = self.crop_image(t2_data, filename=lookup_filename)[0]
                flair = self.crop_image(flair_data, filename=lookup_filename)[0]
            # Second priority: use smart crop if enabled
            elif self.smart_crop:
                print(f"\nSmart crop for {base_filename}")
                tumor_center = self.estimate_tumor_region(t1_data, t1ce_data, t2_data, flair_data)
                t1, crop_coords = self.crop_image(t1_data, tumor_center=tumor_center)
                t1ce = self.crop_image(t1ce_data, tumor_center=tumor_center)[0]
                t2 = self.crop_image(t2_data, tumor_center=tumor_center)[0]
                flair = self.crop_image(flair_data, tumor_center=tumor_center)[0]
            # Last resort: use fixed crop
            else:
                print(f"\nFixed crop for {base_filename}")
                t1, crop_coords = self.crop_image(t1_data)
                t1ce = self.crop_image(t1ce_data)[0]
                t2 = self.crop_image(t2_data)[0]
                flair = self.crop_image(flair_data)[0]

            # Apply normalization
            t1 = self.z_score_normalization(t1)
            t1ce = self.z_score_normalization(t1ce)
            t2 = self.z_score_normalization(t2)
            flair = self.z_score_normalization(flair)

            # Convert to tensors
            t1 = torch.tensor(t1).float().unsqueeze(0)
            t1ce = torch.tensor(t1ce).float().unsqueeze(0)
            t2 = torch.tensor(t2).float().unsqueeze(0)
            flair = torch.tensor(flair).float().unsqueeze(0)

            return t1, t1ce, t2, flair, base_filename, crop_coords


def post_process_prediction(pred_mask, et_threshold=300, min_component_size=10, apply_connected_components=False):
    """Post-process the prediction mask"""
    from scipy import ndimage
    processed_mask = pred_mask.copy()

    if apply_connected_components:
        for label in [1, 2, 3]:
            class_mask = processed_mask == label
            if np.sum(class_mask) > 0:
                labeled_array, num_features = ndimage.label(class_mask)

                if num_features > 0:
                    component_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
                    small_components = component_sizes < min_component_size
                    remove_pixels = small_components[labeled_array - 1]
                    processed_mask[class_mask & remove_pixels] = 0

    # Handle ET regions
    et_mask = processed_mask == 3
    et_voxels = np.sum(et_mask)

    if et_voxels > 0 and et_voxels < et_threshold:
        print(f"Post-processing: ET region found with {et_voxels} voxels (below threshold of {et_threshold})")
        print("Converting ET to NCR/NET")
        processed_mask[et_mask] = 1

    return processed_mask
