import torch
import argparse

import faiss
import faiss.contrib.torch_utils
import numpy as np

import pprint
import itertools
import tqdm
import glob
import os
from torch.utils.data import DataLoader
from cross_sensor import CrossSensorPredictor
from dataloader import make_custom_dataloader
from ben import make_bigearthnet
from model import init_model
from torchvision import transforms

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

def get_features_lists(encoder1, encoder2, data_loader1, data_loader2, dev):
    """Feed all batches from data_loader to backbone and return accumulated features in dictionary with keys "s1" and "s2".

    Args:
        backbone (torch.nn): CSMAE backbone
        data_loader (torch.utils.data.Dataloader): Dataloader
        dev (str): device

    Returns:
        Dict[]: Dictionary with keys "s1" and "s2".
    """
    
    target_list = []
    feat_list_s1 = []
    feat_list_s2 = []

    for (X1, y), (X2, y) in tqdm.tqdm(zip(data_loader1, data_loader2)):
        X1, y = X1.to(dev), y
        X2 = X2.to(dev)
        features_s1 = encoder1(X1).mean(dim=1)
        features_s2 = encoder2(X2).mean(dim=1)

        feat_list_s1.append(features_s1.cpu().detach().numpy())
        feat_list_s2.append(features_s2.cpu().detach().numpy())
        
        target_list.append(y.numpy())

    feat_list_s1 = np.concatenate(feat_list_s1)
    feat_list_s2 = np.concatenate(feat_list_s2)
    target_list = np.concatenate(target_list)

    return feat_list_s1, feat_list_s2, target_list

def main(load_path, device):
    """Performs image retrieval for all uni-/cross-modal cases.

    Args:
        model_id (str): 8-character-id of model name to be evaluated. See under ./trained_models
        device (int): GPU device number
    """

    # initialize paths, devices, aux variables
    dev = f"cuda:{device}"
    torch.cuda.set_device(dev)

    predictor1, encoder1 = init_model(device=device, input_channels = 2)
    predictor2, encoder2 = init_model(device=device, input_channels = 12)
    cross_predictor = CrossSensorPredictor().to(device)

    trainable_modules = ['predictor1', 'predictor2', 'encoder1', 'encoder2', 'cross_predictor']

    # target_encoder = copy.deepcopy(encoder)
    checkpoint = torch.load(load_path, map_location=dev)
    # Load checkpoint
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
    
    from ijepa.src.masks.random import MaskCollator
    mask_collator = MaskCollator()
    generator = torch.Generator(device = 'cpu')
    generator.manual_seed(42)

    collator = MaskCollator()
    _, train_loader1, _ = make_bigearthnet(modality = 1, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=True, transform=transform(1))
    print('Loaded 1')
    _, train_loader2, _ = make_bigearthnet(modality = 2, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=True, transform=transform(2))
    print('Loaded 2')
    # _, test_loader1, _ = make_bigearthnet(modality = 1, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=False, transform=transform(1))
    # print('Loaded 3')
    # _, test_loader2, _ = make_bigearthnet(modality = 2, batch_size = 256, root_path = "/raid/biplab/datasets/BENv1/BENMMfinal/", training=False, transform=transform(2))
    # print('Loaded 4')

    encoder1.eval()
    encoder2.eval()
    encoder1.to(dev)
    encoder2.to(dev)

    # query features from train set / archive features from test set
    with torch.no_grad():
        query_s1, query_s2, query_labels = get_features_lists(encoder1, encoder2, train_loader1, train_loader2, dev)
        archive_s1, archive_s2, archive_labels = get_features_lists(encoder1, encoder2, train_loader1, train_loader2, dev)


    retrieval_dict = {
        'query_s1': query_s1,
        'query_s2': query_s2,
        'archive_s1': archive_s1,
        'archive_s2': archive_s2,
    }

    embed_dim = archive_s1.shape[1]
    print("Archive:", archive_s1.shape, "Labels:", archive_labels.shape)

    k = 10
    index_func = faiss.IndexFlatL2

    faiss.normalize_L2(query_s1)
    faiss.normalize_L2(query_s2)
    faiss.normalize_L2(archive_s1)
    faiss.normalize_L2(archive_s2)

    results = {}
    modalities = ['s1', 's2']
    for from_modality, to_modality in list(itertools.product(modalities, modalities)):

        print(f"Compute IR results for {from_modality}->{to_modality}")

        index = index_func(embed_dim)
        assert index.is_trained, "This index requires training before use."
        index.add(retrieval_dict['archive_' + to_modality])

        # Retrieve k-closest ids
        _, topk_ids_list = index.search(retrieval_dict['query_' + from_modality], k)

        # Convert ids into labels
        topk_labels_list = []
        for topk_ids in topk_ids_list:
            topk_labels = list(map(lambda x: archive_labels[x], topk_ids))
            topk_labels_list.append(topk_labels)

        topk_labels_list = np.asarray(topk_labels_list)

        assert k == topk_labels_list.shape[1]
        num_queries = topk_labels_list.shape[0]

        # Calculate metrics
        precision = .0
        recall = .0
        f1 = .0
        for q, topk_labels in tqdm.tqdm(zip(query_labels, topk_labels_list)):

            # Accumulate per (q, topk-labels) pair
            total_q_prec = .0
            total_q_rec = .0
            total_q_f1 = .0
            print(topk_labels)
            for r in topk_labels:
                print(r)
                num_correct = np.logical_and(q, r).sum()
                prec = num_correct / r.sum()
                rec = num_correct / q.sum()

                total_q_prec += prec
                total_q_rec += rec
                total_q_f1 += (2*prec*rec) / (prec+rec) if num_correct else .0 

            precision += total_q_prec / k
            recall += total_q_rec / k
            f1 += total_q_f1 / k


        avg_precision = precision / num_queries
        avg_recall = recall / num_queries
        avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        results[f"{from_modality}_{to_modality}"] = {
            'prec': avg_precision,
            'rec': avg_recall,
            'f1': avg_f1,
        }

    print("\nRetrieval performance")
    pprint.pprint(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='<8-character-id> of model to be evaluated. Stored under /trained_models/<8-character-id>...')
    parser.add_argument('--device', type=int, help='GPU number')
    
    args = parser.parse_args()

    main(args.path, args.device)
