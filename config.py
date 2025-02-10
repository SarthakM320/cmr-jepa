import argparse
import torch
def get_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training and testing IJEPA on Sentinal1")
    parser.add_argument("--model_name", type=str, default='vit_base', help="Model name")
    parser.add_argument("--pred_depth", type=int, default=12, help="Depth of the predictor")
    parser.add_argument("--pred_emb_dim", type=int, default=384, help="Embedding dimension of the predictor")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--start_lr", type=float, default=0.00005, help="Start Learning rate")
    parser.add_argument("--final_lr", type=float, default=0.001, help="Final Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Weight decay")
    parser.add_argument("--final_weight_decay", type=float, default=0.4, help="Final Weight decay")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--warmup", type=int, default=40, help="Warmup")
    parser.add_argument("--ipe_scale", type=int, default=1, help="IPE scale")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--use_bfloat16", type=bool, default=False, help="use bfloat16")

    
    # Mask collator
    parser.add_argument("--num_enc_masks", type=int, default=1, help="Number of encoder masks")
    parser.add_argument("--num_pred_masks", type=int, default=4, help="Number of predictor masks")
    parser.add_argument("--min_keep", type=int, default=10, help="Minimum number of patches to keep")
    parser.add_argument("--allow_overlap", type=bool, default=False, help="Whether to allow overlap between encoder and predictor")
    parser.add_argument("--enc_mask_scale", type=tuple, default=(0.85, 1.0), help="Encoder mask scale")
    parser.add_argument("--pred_mask_scale", type=tuple, default=(0.15, 0.2), help="Predictor mask scale")
    parser.add_argument("--aspect_ratio", type=tuple, default=(0.75, 1.5), help="Aspect ratio")

    # data
    parser.add_argument("--root_path", type=str, default="/raid/biplab/shabnam/SARFoundational/Datasets/BENv1/BENMMfinal", help="Data directory")
    parser.add_argument("--use_color_distortion", type=bool, default="False", help="Color distortion")
    parser.add_argument("--use_horizontal_flip", type=bool, default="False", help="Horizontal flip")
    parser.add_argument("--use_gaussian_blur", type=bool, default="False", help="Gaussian blur")
    parser.add_argument("--color_jitter", type=bool, default="False", help="Color jitter")
    parser.add_argument("--crop_scale", type=tuple, default=(0.3, 0.1), help="Crop scale")
    parser.add_argument("--pin_mem", type=bool, default="False", help="Pin memory")
    parser.add_argument("--copy_data", type=bool, default="False", help="Copy data")

    

    # Device selection (GPU/CPU)
    if torch.cuda.is_available():
        device_default = "cuda:5"  # Use GPU device 0 if available
    else:
        device_default = "cpu"  # Otherwise fallback to CPU

    parser.add_argument("--device", type=str, default=device_default, help="Device to use for computation (e.g., 'cuda:0' or 'cpu')")

    args = parser.parse_args()
    return args