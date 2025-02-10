import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
import torch
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm

def transform(modality):
    if modality == 1:
        num_channels = 2
    else:
        num_channels = 12
    mean = [0.5] * num_channels
    std = [0.5] * num_channels
    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float), 
        transforms.Resize((224, 224)), 
        transforms.Normalize(mean=mean, std=std)  
    ])
    return transform_


class BigEarthNetDataset(Dataset):
    def __init__(self, root_dir, modality, transform=None, image_paths=None, max_imgs=100000):
        self.root_dir = root_dir
        self.modality = modality
        self.transform = transform
        self.max_imgs = max_imgs
        if image_paths:
            self.image_paths = image_paths
        else:
            self.image_paths = self._load_image_paths()
        
        # Load labels and initialize label binarizer
        self.labels, self.label_binarizer = self._load_labels()

    def _load_image_paths(self):
        image_paths = []
        for layer1 in tqdm(os.listdir(self.root_dir)):
            layer1_path = os.path.join(self.root_dir, layer1)
            if os.path.isdir(layer1_path):
                for file in os.listdir(layer1_path):
                    if self.modality == 2:
                        if file.endswith('2_image.npy'):
                            image_paths.append(os.path.join(layer1_path, file))
                            if len(image_paths) >= self.max_imgs:
                                return image_paths
                    else:
                        if file.endswith('1_image.npy'):
                            image_paths.append(os.path.join(layer1_path, file))
                            if len(image_paths) >= self.max_imgs:
                                return image_paths

                            
        return image_paths


    def _load_labels(self):
        labels_dict = {}
        all_unique_labels = set()
        
        # First pass: collect all unique labels
        for layer1 in tqdm(os.listdir(self.root_dir)):
            layer1_path = os.path.join(self.root_dir, layer1)
            if os.path.isdir(layer1_path):
                label_file = os.path.join(layer1_path, 'labels_metadata.json')
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                        all_unique_labels.update(label_data['labels'])
        
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([list(all_unique_labels)])
        
        for layer1 in tqdm(os.listdir(self.root_dir)):
            layer1_path = os.path.join(self.root_dir, layer1)
            if os.path.isdir(layer1_path):
                label_file = os.path.join(layer1_path, 'labels_metadata.json')
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                        binary_label = label_binarizer.transform([label_data['labels']])[0]
                        labels_dict[layer1] = binary_label
        
        return labels_dict, label_binarizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = np.load(img_path)
        
        # Extract directory name from path
        dir_name = os.path.basename(os.path.dirname(img_path))
        
        if dir_name not in self.labels:
            raise KeyError(f"Directory name '{dir_name}' not found in labels")
        
        label = self.labels[dir_name]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
        

def make_bigearthnet(
    modality,
    batch_size,
    root_path,
    transform = None,
    collator=None,
    pin_mem=False,
    num_workers=1,
    world_size=1,
    rank=0,
    training=True,
    drop_last=False
):
    # Create temporary dataset to get all image paths
    temp_dataset = BigEarthNetDataset(root_path, modality, transform)
    
    # Split the image paths into train and test sets
    train_paths, test_paths = train_test_split(
        temp_dataset.image_paths,
        test_size=0.2,
        random_state=42  
    )
    
    # Create the appropriate dataset based on training flag
    if training:
        dataset = BigEarthNetDataset(root_path, modality, transform, image_paths=train_paths)
    else:
        dataset = BigEarthNetDataset(root_path, modality, transform, image_paths=test_paths)
    g = torch.Generator(device='cpu')
    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
        #generator=g
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        generator = g
    )
    #logger.info('BigEarthNet data loader created')

    return dataset, data_loader, dist_sampler

if __name__ == '__main__':
    main_directory = "/raid/biplab/shabnam/SARFoundational/Datasets/BENv1/BENMMfinal/"

    _, test_loader1, _ = make_bigearthnet(
        modality = 1,
        batch_size=32,
        root_path=main_directory,
        transform=transform(1),
        collator=None,
        pin_mem=True,
        num_workers=1,
        world_size=1,
        rank=0,
        training=True,
        drop_last=False
    )
    _, test_loader2, _ = make_bigearthnet(
        modality = 2,
        batch_size=32,
        root_path=main_directory,
        transform=transform(2),
        collator=None,
        pin_mem=True,
        num_workers=4,
        world_size=1,
        rank=0,
        training=False,
        drop_last=False
    )
    print(test_loader1)
    print(test_loader2)

    print("length of test_loader1: ", len(test_loader1))