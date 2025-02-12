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
    
    Args:
        load_path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        encoder: Loaded encoder model
        predictor: Loaded predictor model
    """
    # Initialize models
    predictor1, encoder1 = init_model(device=device, input_channels = 2)
    predictor2, encoder2 = init_model(device=device, input_channels = 12)
    cross_predictor = CrossSensorPredictor().to(device)

    trainable_modules = ['predictor1', 'predictor2', 'encoder1', 'encoder2', 'cross_predictor']

    # target_encoder = copy.deepcopy(encoder)

    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device)
    
    # Clean state dict helper function
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    # Clean and load state dicts
    encoder1_state_dict = clean_state_dict(checkpoint['model_state_dicts']['encoder1'])
    predictor1_state_dict = clean_state_dict(checkpoint['model_state_dicts']['predictor1'])
    encoder2_state_dict = clean_state_dict(checkpoint['model_state_dicts']['encoder2'])
    predictor2_state_dict = clean_state_dict(checkpoint['model_state_dicts']['predictor2'])
    
    encoder1.load_state_dict(encoder1_state_dict)
    predictor1.load_state_dict(predictor1_state_dict)
    encoder2.load_state_dict(encoder2_state_dict)
    predictor2.load_state_dict(predictor2_state_dict)
        
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
                print(f"Original embedding shape: {z.shape}")
            
            # Apply mean pooling over the sequence dimension (dim=1)
            # This converts shape from [batch_size, seq_len, embedding_dim] to [batch_size, embedding_dim]
            z = torch.mean(z, dim=1)
            
            if itr == 0:
                print(f"After pooling shape: {z.shape}")
            
            all_embeddings.append(z)
            all_labels.append(labels)

    try:
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
    except RuntimeError as e:
        # Print shapes for debugging
        shapes = [emb.shape for emb in all_embeddings]
        print("Embedding shapes:", shapes)
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
    print(f"Query labels shape: {query_labels.shape}")
    print(f"Archive labels shape: {archive_labels.shape}")
    print(f"Indices shape: {indices.shape}")
    
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

def main(args):
    # Set device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # Define paths
    # load_path = '/raid/biplab/sarthak/cmr-jepa/checkpoints/best_model_epoch_6_ 3.0266.pth'
    load_path = args.path

    # Load model
    print("Loading model...")
    encoder1, predictor1, encoder2, predictor2 = load_model_checkpoint(load_path, device)
    
    # Create dataloaders
    from ijepa.src.masks.random import MaskCollator
    mask_collator = MaskCollator()
    generator = torch.Generator(device = 'cpu')
    generator.manual_seed(42)

    # _, _, train_loader1, test_loader1, _, _, _, _, train_loder_2, test_loader2, _, _ = make_custom_dataloader(
    #     transform=transform, 
    #     batch_size=256,  
    #     collator=mask_collator, 
    #     pin_mem=0,  
    #     num_workers=0, 
    #     world_size=1, 
    #     rank=0, 
    #     root_path="/raid/biplab/datasets/BENv1/BENMMfinal/", 
    #     training=True, 
    #     copy_data=False, 
    #     drop_last=True
    # )

    collator = MaskCollator()
    _, train_loader1, _ = make_bigearthnet(modality = 1, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=True, transform=transform(1), collator=collator)
    print('Loaded 1')
    _, train_loader2, _ = make_bigearthnet(modality = 2, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=True, transform=transform(2), collator=collator)
    print('Loaded 2')
    _, test_loader1, _ = make_bigearthnet(modality = 1, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=False, transform=transform(1), collator=collator)
    print('Loaded 3')
    _, test_loader2, _ = make_bigearthnet(modality = 2, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=False, transform=transform(2), collator=collator)
    print('Loaded 4')
    

    #mp.set_start_method('spawn')

    # Extract embeddings for query and archive sets
    
    print("Extracting archive embeddings for S1...")
    archive_features_s1, archive_labels_s1 = extract_embeddings(encoder1, predictor1, train_loader1, device)

    print("Extracting archive embeddings for S2...")
    archive_features_s2, archive_labels_s2 = extract_embeddings(encoder2, predictor2, train_loader2, device)

    print("Extracting query embeddings for S1...")
    query_features_s1, query_labels_s1 = extract_embeddings(encoder1, predictor1, test_loader1, device)

    print("Extracting query embeddings for S2...")
    query_features_s2, query_labels_s2 = extract_embeddings(encoder2, predictor2, test_loader2, device)
    
    features_dict = {
        's1->s2':{
            'query': {
                'features': query_features_s1,
                'labels': query_labels_s1
            },
            'archive': {
                'features': archive_features_s2,
                'labels': archive_labels_s2
            }
        },
        's2->s1':{
            'query': {
                'features': query_features_s2,
                'labels': query_labels_s2
            },
            'archive': {
                'features': archive_features_s1,
                'labels': archive_labels_s1
                }
        }
    }

    print('MultiModal')

    for modality in features_dict.keys():
        query = features_dict[modality]['query']
        archive = features_dict[modality]['archive']

        # Compute nearest neighbors
        k = 5
        print(f"Computing {k}-nearest neighbors...")
        distances, indices = compute_knn(query['features'], archive['features'], k)
        
        # Calculate F1 score
        print("Calculating F1 score...")
        avg_f1 = calculate_f1(query['labels'], archive['labels'], indices, k)
        
        # Print results
        print("\nRetrieval Results:")
        print("------------------")
        print(f"Modality: {modality}")
        print("Model - ", load_path)
        print(f"Mean F1 Score: {(avg_f1):.4f}")


    features_dict = {
        's1':{
            'query': {
                'features': query_features_s1,
                'labels': query_labels_s1
            },
            'archive': {
                'features': archive_features_s1,
                'labels': archive_labels_s1
            }
        },
        's2':{
            'query': {
                'features': query_features_s2,
                'labels': query_labels_s2
            },
            'archive': {
                'features': archive_features_s2,
                'labels': archive_labels_s2
                }
        }
    }
    print('UniModal')

    for modality in features_dict.keys():
        query = features_dict[modality]['query']
        archive = features_dict[modality]['archive']

        # Compute nearest neighbors
        k = 5
        print(f"Computing {k}-nearest neighbors...")
        distances, indices = compute_knn(query['features'], archive['features'], k)
        
        # Calculate F1 score
        print("Calculating F1 score...")
        avg_f1 = calculate_f1(query['labels'], archive['labels'], indices, k)
        
        # Print results
        print("\nRetrieval Results:")
        print("------------------")
        print(f"Modality: {modality}")
        print("Model - ", load_path)
        print(f"Mean F1 Score: {(avg_f1):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    main(parser.parse_args())
