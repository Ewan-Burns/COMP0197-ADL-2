import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.weakly_supervised.resnet import MultiHeadResNet, generate_cams
from src.utils.loss import sec_loss
from src.utils.dataset import TrainTestSplit
from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.utils.random_utils import set_seed, worker_init_fn

# Set fixed random seed
SEED = 42
set_seed(SEED)

def train_with_weights(alpha, beta, gamma, num_epochs=5, lr=1e-4, batch_size=16):
    """Train the weakly-supervised model with specific loss component weights."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    model = MultiHeadResNet(num_classes=3).to(device)
    
    # Dataset setup - use the MultiTargetOxfordPet class from the repository
    dataset = MultiTargetOxfordPet()
    train_set, val_set = TrainTestSplit(dataset, 0.8)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Save path with weight information
    save_path = f"./models/ablation_ep{num_epochs}_a{alpha}_b{beta}_g{gamma}.pth"
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)", leave=False)
        
        for imgs, masks, *_ in bar:  # Use `*_` to ignore any additional outputs
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            # For MultiTargetOxfordPet, the second output is the mask
            masks = masks.to(device)
            
            # Extract class labels from the mask - taking the mode of non-zero values
            # Mask values: 0=background, 1=cat, 2=dog
            # Get the most common class in the mask excluding background (0)
            masks_flat = masks.view(masks.size(0), -1)
            # Filter out background pixels (0) for each image in the batch
            labels = []
            for i in range(masks_flat.size(0)):
                # Get non-zero values (non-background)
                non_zero = masks_flat[i][masks_flat[i] > 0]
                if len(non_zero) > 0:
                    # Get most common class
                    vals, counts = torch.unique(non_zero, return_counts=True)
                    # Convert to 0-based class index (1=cat becomes 0, 2=dog becomes 1)
                    label = vals[counts.argmax()].item() - 1
                else:
                    # If no foreground pixels, default to first class (0)
                    label = 0
                labels.append(label)
            
            # Convert to tensor
            labels = torch.tensor(labels, device=device).long()
            
            feats, segm, cl = model(imgs)
            probs = torch.softmax(segm, dim=1)
            probs = F.interpolate(probs, size=(224, 224), mode="bilinear", align_corners=False)
            
            # CAM seeds
            fc_weights = model.class_head.weight.detach()
            class_ids = labels.to(device)
            
            cams = generate_cams(feats, fc_weights, class_ids)
            
            # Resize CAMs to match seg_head output
            seed_masks = torch.nn.functional.interpolate(
                cams, size=probs.shape[2:], mode="bilinear", align_corners=False
            )
            
            # Convert tensor imgs to numpy for CRF
            # Denormalize first using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            denorm_imgs = imgs * std + mean
            imgs_np = (denorm_imgs * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
            
            # Use custom weightings for ablation
            loss = sec_loss(probs, seed_masks, imgs_np, labels, alpha=alpha, beta=beta, gamma=gamma)
            loss += criterion(cl, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)", leave=False)
        
        with torch.no_grad():
            for imgs, masks, *_ in bar:  # Use `*_` to ignore any additional outputs
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                # Extract class labels similar to training
                masks_flat = masks.view(masks.size(0), -1)
                labels = []
                for i in range(masks_flat.size(0)):
                    non_zero = masks_flat[i][masks_flat[i] > 0]
                    if len(non_zero) > 0:
                        vals, counts = torch.unique(non_zero, return_counts=True)
                        label = vals[counts.argmax()].item() - 1
                    else:
                        label = 0
                    labels.append(label)
                labels = torch.tensor(labels, device=device).long()
                
                feats, segm, cl = model(imgs)
                probs = torch.softmax(segm, dim=1)
                probs = F.interpolate(probs, size=(224, 224), mode="bilinear", align_corners=False)
                
                # CAM seeds
                fc_weights = model.class_head.weight.detach()
                class_ids = labels.to(device)
                
                cams = generate_cams(feats, fc_weights, class_ids)
                
                # Resize CAMs
                seed_masks = torch.nn.functional.interpolate(
                    cams, size=probs.shape[2:], mode="bilinear", align_corners=False
                )
                
                # Convert images for CRF - denormalize first
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                denorm_imgs = imgs * std + mean
                imgs_np = (denorm_imgs * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
                
                loss = sec_loss(probs, seed_masks, imgs_np, labels, alpha=alpha, beta=beta, gamma=gamma)
                loss += criterion(cl, labels)
                
                val_loss += loss.item()
                bar.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    return model, save_path, history


def run_ablation_study(num_epochs=5, lr=1e-4, batch_size=16, seed=42):
    """Run a complete ablation study with different loss component weights."""
    set_seed(seed)
    # Configuration for different ablation experiments
    ablation_configs = [
        {"name": "baseline", "alpha": 1.0, "beta": 1.0, "gamma": 0.5},
        {"name": "no_seed", "alpha": 0.0, "beta": 1.0, "gamma": 0.5},
        {"name": "no_expand", "alpha": 1.0, "beta": 0.0, "gamma": 0.5},
        {"name": "no_constrain", "alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        {"name": "high_seed", "alpha": 2.0, "beta": 1.0, "gamma": 0.5},
        {"name": "high_expand", "alpha": 1.0, "beta": 2.0, "gamma": 0.5},
        {"name": "high_constrain", "alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    ]
    
    results = {}
    training_histories = {}
    
    for config in ablation_configs:
        print(f"\n=== Running ablation: {config['name']} ===")
        print(f"Alpha: {config['alpha']}, Beta: {config['beta']}, Gamma: {config['gamma']}")
        
        # Train model with specific weights
        _, save_path, history = train_with_weights(
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size
        )
        
        # Store training history
        training_histories[config['name']] = history
        
        # Evaluate model
        from src.ablation_evaluator import evaluate_model
        miou, mdice = evaluate_model(save_path)
        
        results[config['name']] = {
            "alpha": config['alpha'],
            "beta": config['beta'],
            "gamma": config['gamma'],
            "miou": miou,
            "mdice": mdice,
            "model_path": save_path
        }
        
        print(f"Results for {config['name']}:")
        print(f"  mIoU: {miou:.4f}")
        print(f"  mDice: {mdice:.4f}")
    
    # Save results to a simple text file
    results_file = './results/ablation_results.txt'
    with open(results_file, 'w') as f:
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  alpha: {result['alpha']}\n")
            f.write(f"  beta: {result['beta']}\n")
            f.write(f"  gamma: {result['gamma']}\n")
            f.write(f"  miou: {result['miou']}\n")
            f.write(f"  mdice: {result['mdice']}\n")
            f.write(f"  model_path: {result['model_path']}\n\n")
    
    # Plot training curves for each configuration
    for name, history in training_histories.items():
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curves for {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'./results/plots/{name}_loss.png')
        plt.close()
    
    # Print summary of all results
    print("\n=== Ablation Study Results ===")
    for name, result in results.items():
        print(f"{name}: mIoU={result['miou']:.4f}, mDice={result['mdice']:.4f}")
    
    return results, training_histories