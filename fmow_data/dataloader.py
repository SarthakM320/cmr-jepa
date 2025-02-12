import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from PIL import Image
import json
from typing import Any, Optional, List

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
        :param csv_path: csv_path (string): path to csv file.
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
    
class FmowRGBRetrieval(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, json_path, transform, is_train=True):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
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
                 label_type: str = 'one_hot',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

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
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

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