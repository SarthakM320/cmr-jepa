import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T

logger = getLogger()

def dims_change(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.tensor(img).unsqueeze(0).to(device)
    img = img.float()
    img = img.permute(0, 3, 1, 2)  # change it to (batch, channels, height, width)
    resize = T.Resize((224, 224))  # Define the resize transformation
    img = resize(img)
    img = img.squeeze(0)
    return img


class NpyImageDataset(Dataset):
    def __init__(self, root, transform=None, limit_folders=100000, sentinel = 1):
        """
        NpyImageDataset for loading .npy images from folders, limited to a certain number of folders.
        
        :param root: Root directory containing folders of .npy files
        :param transform: Transformations to apply to the images
        :param limit_folders: Limit the dataset to the first 'limit_folders' folders
        """
        self.root = root
        self.transform = transform
        self.limit_folders = limit_folders
        self.sentinel = sentinel

        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load .npy file paths from the first 'limit_folders' folders.
        """
        samples = []
        folder_count = 0

        for root_dir, dirs, files in os.walk(self.root):
            if folder_count >= self.limit_folders:
                break  # Stop once we reach the limit of folders
            
            if files:
                folder_count += 1  # Count the folders with .npy files

                for file in files:
                    if(self.sentinel == 2):
                        if file.endswith('2_image.npy'):
                            samples.append(os.path.join(root_dir, file))
                    else:
                        if file.endswith('1_image.npy'):
                            samples.append(os.path.join(root_dir, file))

        logger.info(f'Loaded {len(samples)} .npy files from the first {self.limit_folders} folders in {self.root}')
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        npy_path = self.samples[index]
        img = np.load(npy_path)  # Load the .npy file
        img = dims_change(img)  # Change dimensions and resize the image
            
        return img  # No target, just returning the image


def make_custom_dataloader(
    transform,
    batch_size,
    collator=None,
    pin_mem=False,
    num_workers=10,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    split_ratio=0.2  # New split_ratio parameter
):
    """
    Custom DataLoader function with train-test split for the .npy dataset structure.
    
    :param transform: Transformations to apply to the dataset images
    :param batch_size: Number of samples per batch
    :param collator: Function to collate samples into batches
    :param pin_mem: Whether to use pinned memory
    :param num_workers: Number of worker threads for data loading
    :param world_size: Number of distributed processes (for DistributedSampler)
    :param rank: Rank of the current process (for DistributedSampler)
    :param root_path: Path to the root directory containing the dataset
    :param training: Whether to load training or validation data
    :param copy_data: Option to copy data locally (not used in this version)
    :param drop_last: Whether to drop the last incomplete batch
    :param subset_file: Optional file to filter the dataset (not used here)
    :param split_ratio: Proportion of the data to allocate to the test set
    """
    # Load the dataset
    dataset_1 = NpyImageDataset(
        root=root_path,
        transform=transform
    )
    dataset_2 = NpyImageDataset(
        root=root_path,
        transform=transform,
        sentinel=2
    )

    if subset_file is not None:
        dataset_1= ImageNetSubset(dataset_1, subset_file)
        dataset_2 = ImageNetSubset(dataset_2, subset_file)
    
    logger.info('Npy dataset created')

    # Calculate the train and test sizes
    dataset_size = len(dataset_1)
    test_size = int(split_ratio * dataset_size)
    train_size = dataset_size - test_size

    # Split the dataset into train and test sets
    train_dataset1, test_dataset1 = random_split(dataset_1, [train_size, test_size])
    train_dataset2, test_dataset2 = random_split(dataset_2, [train_size, test_size])
    logger.info(f'Dataset split: {train_size} training samples, {test_size} test samples')

    # Create distributed samplers for both train and test sets
    train_sampler_1 = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset1,
        num_replicas=world_size,
        rank=rank
    )
    train_sampler_2 = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset2,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler_1 = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset1,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler_2 = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset2,
        num_replicas=world_size,
        rank=rank
    )

    # DataLoaders with the custom collator
    train_loader1 = DataLoader(
        train_dataset1,
        collate_fn=collator,
        sampler=train_sampler_1,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    test_loader1 = DataLoader(
        test_dataset1,
        collate_fn=collator,
        sampler=test_sampler_1,
        batch_size=batch_size,
        drop_last=False,  
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    
    train_loader2 = DataLoader(
        train_dataset2,
        collate_fn=collator,
        sampler=train_sampler_2,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    test_loader2 = DataLoader(
        test_dataset2,
        collate_fn=collator,
        sampler=test_sampler_2,
        batch_size=batch_size,
        drop_last=False,  
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    logger.info('Train and test dataloaders created')
    return train_dataset1, test_dataset1, train_loader1, test_loader1, train_sampler_1, test_sampler_1, train_dataset2, test_dataset2, train_loader2, test_loader2, train_sampler_2, test_sampler_2

class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset for filtering samples (not changed).
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                img = line.strip()
                new_samples.append(os.path.join(root, img))
        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.dataset[index]  # Delegate to the parent dataset


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    """
    Copy data locally (same as before).
    """
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Command took {(time.time() - start_time) / 60.0} min.')
            with open(tmp_sgnl_file, 'w') as f:
                f.write('Done copying locally.')
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
