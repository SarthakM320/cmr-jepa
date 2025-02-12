import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json
from typing import Any, Optional, List
import rasterio
from logging import getLogger
import warnings
warnings.filterwarnings("ignore")

logger = getLogger()

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]

class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
    
class FmowRGB(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, json_path, transform, is_train=True):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param json_path: json_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.image_arr = json.load(open(json_path, 'r'))['train' if is_train else 'val']
        # Calculate len

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open 
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)

        return img_as_tensor

    def __len__(self):
        return len(self.image_arr)
    
class FmowRGBRetrieval(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, json_path, transform, is_train=True):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param json_path: json_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.image_arr = json.load(open(json_path, 'r'))['train' if is_train else 'val']
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)

        return img_as_tensor

    def __len__(self):
        return len(self.image_arr)
    
class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class FmowSentinel(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 json_path: str,
                 transform: Any,
                 label_type: str = 'one-hot',is_train=True):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param json_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        """
        super().__init__(in_c=13)
        # Filter by category
        self.categories = CATEGORIES
        

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type
        self.image_arr = json.load(open(json_path, 'r'))['train' if is_train else 'val']
        self.transforms = transform

    def __len__(self):
        return len(self.image_arr)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.image_arr[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection)  # (h, w, c)
        

        img_as_tensor = self.transforms(images)  # (c, h, w)
       
        return img_as_tensor

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)
    
def build_fmow_dataset(dataset_type, is_train) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided 
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    json_path = f'fmow_data/{dataset_type}.json'

    if dataset_type == 'fmow_rgb':
        mean = FmowRGB.mean
        std = FmowRGB.std
        transform = FmowRGB.build_transform(is_train, 224, mean, std)
        dataset = FmowRGB(json_path, transform, is_train=is_train)
    elif dataset_type == 'fmow_sentinel':
        mean = FmowSentinel.mean
        std = FmowSentinel.std
        transform = FmowSentinel.build_transform(is_train, 224, mean, std)
        dataset = FmowSentinel(json_path, transform, is_train = is_train)
   
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    return dataset


def make_custom_dataloader(
    transform,
    batch_size,
    collator=None,
    pin_mem=False,
    num_workers=10,
    drop_last=True
):
    fmow_rgb_train = build_fmow_dataset('fmow_rgb', is_train=True)
    fmow_sentinel_train = build_fmow_dataset('fmow_sentinel', is_train=True)
    fmow_rgb_test = build_fmow_dataset('fmow_rgb', is_train=False)
    fmow_sentinel_test = build_fmow_dataset('fmow_sentinel', is_train=False)

    logger.info('FMOW dataset created')
    logger.info(f'FMOW RGB: {len(fmow_rgb_train)} training samples, {len(fmow_rgb_test)} test samples')
    logger.info(f'FMOW SENTINEL: {len(fmow_sentinel_train)} training samples, {len(fmow_sentinel_test)} test samples')

    train_loader1 = DataLoader(
        fmow_rgb_train,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    test_loader1 = DataLoader(
        fmow_rgb_test,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=False,  
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    
    train_loader2 = DataLoader(
        fmow_sentinel_train,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    test_loader2 = DataLoader(
        fmow_sentinel_test,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=False,  
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('Train and test dataloaders created')

    return fmow_rgb_train, fmow_rgb_test, train_loader1, test_loader1, None, None, fmow_sentinel_train, fmow_sentinel_test, train_loader2, test_loader2, None, None

