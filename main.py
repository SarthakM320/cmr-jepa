import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from logging import getLogger, StreamHandler, FileHandler, INFO
from model import init_model, VisionTransformer
from cross_sensor import CrossSensorPredictor
from vicreg import vicreg_loss as vicregloss
from ijepa.src.masks.utils import apply_masks
from ijepa.src.masks.random import MaskCollator 
from ijepa.src.transforms import make_transforms
from config import get_arguments
from dataloader import make_custom_dataloader
import os
import wandb
import numpy as np
import random

logger = getLogger()
logger.setLevel(INFO)
handler = StreamHandler()
handler.setLevel(INFO)
logger.addHandler(handler)

# File handler for logging
file_handler = FileHandler('training.log')
file_handler.setLevel(INFO)
logger.addHandler(file_handler)

def set_seed(seed: int = 42):
    """Set seed for reproducibility across multiple frameworks."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all GPUs)
    
    
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.benchmark = False  # Disable optimizations for reproducibility
    
    print(f"Seed set to {seed}")

def main():
    set_seed(42)
    args = get_arguments()
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    wandb.init(project="CMR_Jepa", config=args)
    wandb.config.update(args)

    main_directory = "/raid/biplab/datasets/BENv1/BENMMfinal/"
    mask_collator = MaskCollator(
        input_size=args.img_size,
        patch_size=args.patch_size,
    )
    
    transform = make_transforms(
        crop_size=args.img_size,
        crop_scale=args.crop_scale,
        gaussian_blur=args.use_gaussian_blur,
        horizontal_flip=args.use_horizontal_flip,
        color_distortion=args.use_color_distortion,
        color_jitter=args.color_jitter
    )

    train_dataset1, test_dataset1, train_loader1, test_loader1, train_sampler1, test_sampler1, train_dataset2, test_dataset2, train_loader2, test_loader2, train_sampler2, test_sampler2 = make_custom_dataloader(
        transform=transform, 
        batch_size=args.batch_size,  
        collator=mask_collator, 
        pin_mem=0,  
        num_workers=0, 
        world_size=1, 
        rank=0, 
        root_path=main_directory, 
        training=True, 
        copy_data=False, 
        drop_last=True
    )

    # Initialize models
    # encoder1 = VisionTransformer(in_chans=2).to(device)
    # encoder2 = VisionTransformer(in_chans=12).to(device)
    predictor1, encoder1 = init_model(img_size=args.img_size, device=device, input_channels=2)
    predictor2, encoder2 = init_model(img_size=args.img_size, device=device, input_channels=12)
    
    target_encoder1 = VisionTransformer(in_chans=12).to(device)
    target_encoder2 = VisionTransformer(in_chans=2).to(device)
    cross_predictor = CrossSensorPredictor().to(device)

    trainable_modules = ['predictor1', 'predictor2', 'encoder1', 'encoder2', 'cross_predictor']
    all_modules = trainable_modules + ['target_encoder1', 'target_encoder2']

    for p in target_encoder1.parameters():
        p.requires_grad = False
    
    for p in target_encoder2.parameters():
        p.requires_grad = False

    ipe = len(train_loader1)
    ema = [0.996, 1]
   
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.num_epochs*args.ipe_scale)
                          for i in range(int(ipe*args.num_epochs*args.ipe_scale)+1))
    

    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = []
    for module in trainable_modules:
        try:
            params.extend(list(locals().get (module).parameters()))
        except:
            print(module)

    optimizer = torch.optim.AdamW(params, lr=args.start_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    scaler = GradScaler(enabled=not args.use_bfloat16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs * len(train_loader1), 
        eta_min=args.final_lr
    )
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            for module in all_modules:
                locals().get(module).load_state_dict(checkpoint['model_state_dicts'][module])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed training from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint {args.resume_checkpoint} not found. Starting from scratch.")

    best_loss = float('inf')
    best_epoch = 0
    
    # Create checkpoint directory
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (batch1, batch2) in enumerate(zip(train_loader1, train_loader2)):
            # Unpack both batches
            udata1, masks_enc, masks_pred = batch1
            udata2, _, _ = batch2 

            x1 = udata1.to(device)
            x2 = udata2.to(device)
            masks_1 = masks_enc[0].to(device)
            masks_2 = masks_pred[0].to(device)
            optimizer.zero_grad()


            with autocast(enabled=not args.use_bfloat16):
                z1 = encoder1(x1, masks_1)
                z2 = encoder2(x2, masks_1)
                z1_pred = predictor1(z1, masks_1, masks_2)
                z2_pred = predictor2(z2, masks_1, masks_2)
                z1_target = target_encoder2(x1)
                z2_target = target_encoder1(x2)
                z1_target = F.layer_norm(z1_target, (z1_target.size(-1),))
                z1_target = apply_masks(z1_target, masks_2)
                z2_target = F.layer_norm(z2_target, (z2_target.size(-1),))
                z2_target = apply_masks(z2_target, masks_2)
                z1_cross, z2_cross = cross_predictor(z1_pred, z2_pred) # contrastive

                vicreg_loss = vicregloss(z1.mean(dim = 1), z2.mean(dim = 1)) 
                pred_loss = nn.MSELoss()(z1_pred, z2_target) + nn.MSELoss()(z2_pred, z1_target) # TODO
            
            loss = vicreg_loss * 0.1 + pred_loss
            wandb.log({'train/pred_loss': pred_loss, 'train/vicreg_loss': vicreg_loss, 'train/total_loss': loss})

            if not torch.isfinite(loss).all():
                logger.error(f"NaN or Inf found in 'loss' at iteration {batch_idx}")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(encoder2.parameters(), target_encoder1.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                for param_q, param_k in zip(encoder1.parameters(), target_encoder2.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                # encoder_2 -> target_encoder1
                # encoder_1 -> target_encoder2

            import json
            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "iteration": batch_idx + 1})
            if batch_idx %10 == 0:
                logger.info(json.dumps({"train_loss": loss.item(), "epoch": epoch + 1, "iteration": batch_idx + 1}))


        for batch_idx, (batch1, batch2) in enumerate(zip(test_loader1, test_loader2)):
            # Unpack both batches
            udata1, masks_enc, masks_pred = batch1
            udata2, _, _ = batch2 
            x1 = udata1.to(device)
            x2 = udata2.to(device)
            masks_1 = masks_enc[0].to(device)
            masks_2 = masks_pred[0].to(device)
            with torch.no_grad():
                z1 = encoder1(x1, masks_1)
                z2 = encoder2(x2, masks_1)
                z1_pred = predictor1(z1, masks_1, masks_2)
                z2_pred = predictor2(z2, masks_1, masks_2)
                z1_target = target_encoder2(x1)
                z2_target = target_encoder1(x2)
                z1_target = F.layer_norm(z1_target, (z1_target.size(-1),))
                z1_target = apply_masks(z1_target, masks_2)
                z2_target = F.layer_norm(z2_target, (z2_target.size(-1),))
                z2_target = apply_masks(z2_target, masks_2)
                z1_cross, z2_cross = cross_predictor(z1_pred, z2_pred)

                vicreg_loss = vicregloss(z1.mean(dim = 1),z2.mean(dim = 1)) 
                pred_loss = nn.MSELoss()(z1_cross.squeeze(0), z2_target.squeeze(0)) + nn.MSELoss()(z2_cross.squeeze(0), z1_target.squeeze(0)) # TODO
            
            loss = vicreg_loss * 0.1 + pred_loss
            wandb.log({"val_loss": loss.item(), "epoch": epoch + 1, "iteration": batch_idx + 1})
            total_loss += loss.item() * x1.size(0)
            total_samples += x1.size(0)

        avg_loss = total_loss / total_samples
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Val Loss: {avg_loss:.4f}")

        model_state_dicts = {}
        for module in all_modules:
            if module in locals():
                model_state_dicts[module] = locals()[module].state_dict()
            else:
                print(f"WARNING: Module '{module}' not found in locals(). Skipping...")

        if avg_loss < best_loss or avg_loss == best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dicts': model_state_dicts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_loss
            }, os.path.join(save_dir, f'best_model_epoch_{epoch+1}_{best_loss : .4f}.pth'))
            logger.info(f"New best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

            # Save model weights to W&B
            wandb.save(os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth'))

        # Learning rate scheduling
        scheduler.step()

    logger.info(f"Best Model Achieved at Epoch {best_epoch} with Loss {best_loss:.4f}")

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    main()
