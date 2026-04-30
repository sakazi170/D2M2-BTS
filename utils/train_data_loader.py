import os
import torch
import numpy as np
from monai.data import Dataset
from monai.transforms import (
    LoadImaged,
    SpatialPadd,
    Compose,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAxisFlipd,
    MapLabelValued,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
)
from monai.transforms import Transform
from monai.config import KeysCollection


class CustomRandomCropd(Transform):
    """Random crop for BraTS volumetric data."""

    def __init__(self, keys: KeysCollection):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        top_crop    = np.random.randint(30, 90)
        left_crop   = np.random.randint(40, 80)
        z_crop      = 16
        bottom_crop = top_crop  + 128
        right_crop  = left_crop + 128
        z_end       = z_crop    + 128

        d = dict(data)
        for key in self.keys:
            d[key] = d[key][
                :,
                top_crop:bottom_crop,
                left_crop:right_crop,
                z_crop:z_end]
        return d


class ConvertToMultiChannelBraTS(Transform):
    """
    Convert single-channel integer label map to 3 binary channel masks.

    BraTS label values (after remapping for 2019/2020):
        0 = background
        1 = NCR  (necrotic core)
        2 = ED   (edema)
        3 = ET   (enhancing tumor)

    Output channels: [ET, ED, NCR] — matches model output order.
    Shape: (1, H, W, D) → (3, H, W, D)
    """

    def __init__(self, keys: KeysCollection):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]               # (1, H, W, D) float or long
            d[key] = torch.cat([
                (label == 3).float(),    # channel 0: ET
                (label == 2).float(),    # channel 1: ED
                (label == 1).float(),    # channel 2: NCR
            ], dim=0)                    # (3, H, W, D)
        return d


class SubjectReader:
    def __init__(self, train_dir, train_dataset, training_size=None):
        self.train_dir      = train_dir
        self.train_dataset  = train_dataset
        self.training_size  = training_size
        self.train_subjects = os.listdir(self.train_dir)

    def get_subjects(self, subject_list, data_dir):
        subjects = []
        for subject_name in subject_list:
            try:
                if self.train_dataset in ['brats2019', 'brats2020']:
                    subject = {
                        't1'   : os.path.join(data_dir, subject_name, f'{subject_name}_t1.nii.gz'),
                        't1ce' : os.path.join(data_dir, subject_name, f'{subject_name}_t1ce.nii.gz'),
                        't2'   : os.path.join(data_dir, subject_name, f'{subject_name}_t2.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}_flair.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}_seg.nii.gz'),
                        'name' : subject_name,
                    }
                elif self.train_dataset in ['brats2021', 'brats2023']:
                    subject = {
                        't1'   : os.path.join(data_dir, subject_name, f'{subject_name}-t1n.nii.gz'),
                        't1ce' : os.path.join(data_dir, subject_name, f'{subject_name}-t1c.nii.gz'),
                        't2'   : os.path.join(data_dir, subject_name, f'{subject_name}-t2w.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}-t2f.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}-seg.nii.gz'),
                        'name' : subject_name,
                    }
                else:
                    raise ValueError(f'Unsupported dataset: {self.train_dataset}')

                if all(os.path.exists(v) for v in subject.values()
                       if isinstance(v, str) and v.endswith('.nii.gz')):
                    subjects.append(subject)
                else:
                    print(f'Warning: missing files for {subject_name}, skipping.')

            except Exception as e:
                print(f'Error processing {subject_name}: {e}')

        return subjects

    def get_trainset(self):
        transform     = self.get_training_transform()
        train_subjects = self.get_subjects(self.train_subjects, self.train_dir)
        trainset      = Dataset(data=train_subjects, transform=transform)
        print(f'Training dataset ready. Length: {len(trainset)}')
        return trainset

    def get_training_transform(self):
        training_keys = ('t1', 't1ce', 't2', 'flair', 'label')
        image_keys    = ('t1', 't1ce', 't2', 'flair')

        transform_list = [
            LoadImaged(keys=training_keys),
            # EnsureChannelFirstd replaces deprecated AddChanneld
            EnsureChannelFirstd(keys=training_keys),
        ]

        # label remapping: BraTS 2019/2020 use label=4 for ET → remap to 3
        if self.train_dataset in ['brats2019', 'brats2020']:
            transform_list.append(
                MapLabelValued(
                    keys='label',
                    orig_labels=(0, 1, 2, 4),
                    target_labels=(0, 1, 2, 3)))

        transform_list.extend([
            SpatialPadd(keys=training_keys, spatial_size=(240, 240, 160)),
            CustomRandomCropd(keys=training_keys),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            RandAxisFlipd(keys=training_keys, prob=0.5),
            RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
            EnsureTyped(keys=training_keys),
            # convert label (1, H, W, D) → (3, H, W, D) binary masks [ET, ED, NCR]
            ConvertToMultiChannelBraTS(keys=('label',)),
        ])

        return Compose(transform_list)