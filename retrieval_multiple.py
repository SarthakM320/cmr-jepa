import os
import sys
from torchvision import transforms
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from model import init_model
from tqdm import tqdm
import torch.multiprocessing as mp
from cross_sensor import CrossSensorPredictor
from dataloader import make_custom_dataloader
from tqdm import tqdm
from ben import make_bigearthnet
from ijepa.src.masks.random import MaskCollator
from logging import getLogger, StreamHandler, FileHandler, INFO

logger = getLogger()
logger.setLevel(INFO)
handler = StreamHandler()
handler.setLevel(INFO)
logger.addHandler(handler)

# File handler for logging
file_handler = FileHandler('retrieval_outputs/random_all.log')
file_handler.setLevel(INFO)
logger.addHandler(file_handler)

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

# Model configuration parameters
MODEL_CONFIG = {
    'wd': 0.04,
    'final_wd': 0.4,
    'start_lr': 0.0002,
    'lr': 0.001,
    'final_lr': 1e-6,
    'ipe': 1000,
    'warmup': 40,
    'num_epochs': 50,
    'ipe_scale': 1,
    'use_bfloat16': True
}

def load_model_checkpoint(load_path, device):    
    """
    Load model and checkpoint from the given path.
    """
    predictor1, encoder1 = init_model(device=device, input_channels=2)
    predictor2, encoder2 = init_model(device=device, input_channels=12)
    cross_predictor = CrossSensorPredictor().to(device)

    checkpoint = torch.load(load_path, map_location=device)
    
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    encoder1.load_state_dict(clean_state_dict(checkpoint['model_state_dicts']['encoder1']))
    predictor1.load_state_dict(clean_state_dict(checkpoint['model_state_dicts']['predictor1']))
    encoder2.load_state_dict(clean_state_dict(checkpoint['model_state_dicts']['encoder2']))
    predictor2.load_state_dict(clean_state_dict(checkpoint['model_state_dicts']['predictor2']))
        
    return encoder1, predictor1, encoder2, predictor2

def extract_embeddings(encoder, predictor, data_loader, device):
    """
    Extract embeddings from the model for all images in the dataloader.
    Handles variable sequence lengths by applying mean pooling over the sequence dimension.
    """
    encoder.eval()
    predictor.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for itr, (udata, masks_enc, masks_pred) in enumerate(tqdm(data_loader)):
            images, labels = udata
            
            images = images.to(device)
            labels = labels.to(device)
            masks_enc = [mask.to(device) for mask in masks_enc]
            masks_pred = [mask.to(device) for mask in masks_pred]
            
            # Get embeddings
            # z = encoder(images, masks_enc)
            # z = predictor(z, masks_enc, masks_pred)
            z = encoder(images)
            
            if itr == 0:
                logger.info(f"Original embedding shape: {z.shape}")
            
            # Apply mean pooling over the sequence dimension (dim=1)
            # This converts shape from [batch_size, seq_len, embedding_dim] to [batch_size, embedding_dim]
            z = torch.mean(z, dim=1)
            
            if itr == 0:
                logger.info(f"After pooling shape: {z.shape}")
            
            all_embeddings.append(z)
            all_labels.append(labels)

    try:
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
    except RuntimeError as e:
        # logger.info shapes for debugging
        shapes = [emb.shape for emb in all_embeddings]
        logger.info("Embedding shapes:", shapes)
        raise e
    
    return embeddings, labels

def compute_knn(query_feats, archive_feats, k=10):
    """
    Compute k-nearest neighbors using cosine similarity.
    Returns indices that are guaranteed to be within bounds of archive_feats.
    """
    # Convert to numpy if needed
    if torch.is_tensor(query_feats):
        query_feats = query_feats.cpu().numpy()
    if torch.is_tensor(archive_feats):
        archive_feats = archive_feats.cpu().numpy()
        
    # Normalize features
    query_norm = query_feats / (np.linalg.norm(query_feats, axis=1)[:, np.newaxis] + 1e-8)
    archive_norm = archive_feats / (np.linalg.norm(archive_feats, axis=1)[:, np.newaxis] + 1e-8)
    
    # Ensure k doesn't exceed the number of archive samples
    k = min(k, len(archive_norm))
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nbrs.fit(archive_norm)
    distances, indices = nbrs.kneighbors(query_norm)
    
    # Verify indices are within bounds
    max_idx = len(archive_feats) - 1
    indices = np.clip(indices, 0, max_idx)
    
    return distances, indices

def calculate_f1(query_labels, archive_labels, indices, k):
    """Calculate F1 score for retrieval results."""
    if torch.is_tensor(query_labels):
        query_labels = query_labels.cpu().numpy()
    if torch.is_tensor(archive_labels):
        archive_labels = archive_labels.cpu().numpy()
    
    # Verify shapes before proceeding
    logger.info(f"Query labels shape: {query_labels.shape}")
    logger.info(f"Archive labels shape: {archive_labels.shape}")
    logger.info(f"Indices shape: {indices.shape}")
    
    num_queries = len(query_labels)
    f1_scores = []
    
    for i in range(num_queries):
        query_label = query_labels[i]
        # Ensure indices are within bounds
        valid_indices = np.clip(indices[i], 0, len(archive_labels) - 1)
        retrieved_labels = archive_labels[valid_indices]
        
        # Calculate F1 score for each retrieved neighbor
        neighbor_f1_scores = []
        for retrieved_label in retrieved_labels:
            f1 = f1_score(query_label, retrieved_label, average='macro')
            neighbor_f1_scores.append(f1)
        
        # Average F1 score over k neighbors
        avg_neighbor_f1 = np.mean(neighbor_f1_scores)
        f1_scores.append(avg_neighbor_f1)

    return np.mean(f1_scores)

def main(args, paths):
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logger.info(device)
    collator = MaskCollator()
    
    _, train_loader1, _ = make_bigearthnet(modality=1, batch_size=256, root_path=args.root_path, training=True, transform=transform(1), collator=collator)
    _, train_loader2, _ = make_bigearthnet(modality=2, batch_size=256, root_path=args.root_path, training=True, transform=transform(2), collator=collator)
    _, test_loader1, _ = make_bigearthnet(modality=1, batch_size=256, root_path=args.root_path, training=False, transform=transform(1), collator=collator)
    _, test_loader2, _ = make_bigearthnet(modality=2, batch_size=256, root_path=args.root_path, training=False, transform=transform(2), collator=collator)
    
    for load_path in paths:
        logger.info(f"Testing model: {load_path}")
        encoder1, predictor1, encoder2, predictor2 = load_model_checkpoint(load_path, device)
        
        logger.info("Extracting embeddings for training and testing sets...")
        archive_features_s1, archive_labels_s1 = extract_embeddings(encoder1, predictor1, train_loader1, device)
        archive_features_s2, archive_labels_s2 = extract_embeddings(encoder2, predictor2, train_loader2, device)
        query_features_s1, query_labels_s1 = extract_embeddings(encoder1, predictor1, test_loader1, device)
        query_features_s2, query_labels_s2 = extract_embeddings(encoder2, predictor2, test_loader2, device)
        
        for modality, (query_feats, query_labels, archive_feats, archive_labels) in {
            's1->s2': (query_features_s1, query_labels_s1, archive_features_s2, archive_labels_s2),
            's2->s1': (query_features_s2, query_labels_s2, archive_features_s1, archive_labels_s1)
        }.items():
            logger.info(f"Computing retrieval results for {modality}")
            distances, indices = compute_knn(query_feats, archive_feats, k=5)
            avg_f1 = calculate_f1(query_labels, archive_labels, indices, k=5)
            logger.info(f"{modality} Mean F1 Score: {avg_f1:.4f}")

        for modality, (query_feats, query_labels, archive_feats, archive_labels) in {
            's1->s1': (query_features_s1, query_labels_s1, archive_features_s1, archive_labels_s1),
            's2->s2': (query_features_s2, query_labels_s2, archive_features_s2, archive_labels_s2)
        }.items():
            logger.info(f"Computing retrieval results for {modality}")
            distances, indices = compute_knn(query_feats, archive_feats, k=5)
            avg_f1 = calculate_f1(query_labels, archive_labels, indices, k=5)
            logger.info(f"{modality} Mean F1 Score: {avg_f1:.4f}")

    logger.info('\n\n')

        
if __name__ == "__main__":
    import re
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)_', filename)
        return int(match.group(1)) if match else float('inf')  # Default to high number if not found

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='List of checkpoint paths')
    parser.add_argument('--root_path', help='Root path for dataset', default='/raid/biplab/datasets/BENv1/BENMMfinal/')
    args = parser.parse_args()
    paths = sorted([os.path.join(args.path, file) for file in os.listdir(args.path) if file.endswith('.pth')], key=extract_epoch)
    logger.info(args.path)
    logger.info('Starting process')
    logger.info(paths)
    main(args, paths)
