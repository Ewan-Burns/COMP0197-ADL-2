import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse # Import argparse

# Import base dataset class and metrics (though evaluation is removed, visualization needs dataset)
from src.MultiTargetOxfordPet import MultiTargetOxfordPet, OxfordIIITPet
from src.weakly_supervised.resnet import MultiHeadResNet, generate_cams
from src.weakly_supervised.sec import sec_loss
# Use TrainTestSplit from utils.dataset
from src.utils.dataset import TrainTestSplit, ResNetTransform

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


# Define training function accepting args and device
def train_weakly_supervised(train_loader, args, device, num_classes=3):
    """Trains the weakly supervised MultiHeadResNet model using provided arguments."""
    print(f"Starting Weakly-Supervised Training on {device}...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  SEC Alpha (Seed): {args.sec_alpha}")
    print(f"  SEC Beta (Expand): {args.sec_beta}")
    print(f"  SEC Gamma (Constrain): {args.sec_gamma}")

    model = MultiHeadResNet(num_classes=num_classes).to(device) # Move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # Use lr from args
    classification_criterion = nn.CrossEntropyLoss() # Separate criterion for class head

    for epoch in range(args.epochs): # Use epochs from args
        model.train()
        epoch_loss = 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

        for imgs, labels in bar:
            optimizer.zero_grad()

            imgs = imgs.to(device) # Move data to device
            labels = labels.to(device) # Move data to device

            # Forward pass
            feats, segm, cl = model(imgs) # segm is raw logits

            # --- Calculate SEC Loss ---
            # 1. Get probabilities from segmentation head output
            probs = torch.softmax(segm, dim=1)
            # Upsample if segm_head output is smaller than input
            if probs.shape[2:] != imgs.shape[2:]:
                probs = F.interpolate(
                    probs, size=imgs.shape[2:], mode="bilinear", align_corners=False
                )

            # 2. Generate CAM seeds
            fc_weights = model.class_head.weight.detach() # Weights are on device
            # Pass features and labels (already on device)
            cams = generate_cams(feats.detach(), fc_weights, labels) # Detach feats?

            # Resize CAMs to match probability map size
            seed_masks = F.interpolate(
                cams, size=probs.shape[2:], mode="bilinear", align_corners=False
            )

            # 3. Prepare images for CRF (needs numpy uint8 on CPU)
            # Define mean/std on the correct device
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            imgs_denorm = imgs * std + mean
            # Move to CPU before converting to numpy
            imgs_np = (imgs_denorm.permute(0, 2, 3, 1).cpu() * 255).numpy().astype("uint8")

            # 4. Calculate the combined SEC loss
            # Pass weights from args to sec_loss
            # sec_loss expects integer labels for expand_loss CE part
            sec = sec_loss(
                probs,
                seed_masks,
                imgs_np,
                labels, # Pass integer labels
                alpha=args.sec_alpha,
                beta=args.sec_beta,
                gamma=args.sec_gamma
            )

            # --- Calculate Classification Loss ---
            class_loss = classification_criterion(cl, labels)

            # --- Total Loss ---
            total_loss = sec + class_loss # Combine losses

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            bar.set_postfix(loss=total_loss.item(), sec=sec.item(), cls=class_loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} Average Loss: {avg_epoch_loss:.4f}")

    # Construct model path based on hyperparameters
    model_path = f"./models/weakly_sup_ep{args.epochs}_lr{args.lr}_a{args.sec_alpha}_b{args.sec_beta}_g{args.sec_gamma}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path


# Define visualization function accepting device
def visualize_prediction(model, dataset, num_samples=5, device='cpu'):
    """
    Visualizes input, CAM, and predicted segmentation for a few samples.
    Assumes dataset provides (img, mask) for visualization.
    Moves necessary tensors to the specified device for model inference.
    """
    print(f"Visualizing Predictions on {device}...")
    model.eval()
    model.to(device) # Ensure model is on the correct device
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i in indices:
            # Dataset yields (image, mask) - image is likely already transformed (tensor)
            img, mask = dataset[i] # img/mask are on CPU
            img_input = img.unsqueeze(0).to(device) # Move input image to device

            # Derive image-level label from mask (on CPU) for CAM generation
            present_classes = torch.unique(mask)
            pet_class = present_classes[present_classes > 0]
            if len(pet_class) == 0: continue # Skip if no pet class found
            label = pet_class[0].item() # Integer class label

            # Get model outputs
            feats, segm, _ = model(img_input)
            pred_probs = F.softmax(segm, dim=1) # Output probs are on device

            # Generate CAM for the specific class present
            fc_weights = model.class_head.weight.detach() # Weights are on device
            label_tensor = torch.tensor([label], device=device) # Create label tensor on device
            cam = generate_cams(feats, fc_weights, label_tensor)[0] # CAM is on device (1, H_feat, W_feat)

            # Upsample CAM and prediction to input size
            cam_upsampled = F.interpolate(
                cam.unsqueeze(0), size=img.shape[1:], mode="bilinear", align_corners=False
            )[0] # (1, H, W) - on device
            pred_mask_upsampled = F.interpolate(
                pred_probs, size=img.shape[1:], mode="bilinear", align_corners=False
            )[0] # (C, H, W) - on device

            # --- Plotting ---
            # Denormalize original image tensor (img is on CPU)
            mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std_cpu = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm_cpu = img * std_cpu + mean_cpu
            img_np = img_denorm_cpu.permute(1, 2, 0).cpu().numpy() # Convert to numpy for display
            img_np = np.clip(img_np, 0, 1)

            # Move CAM and prediction to CPU for numpy conversion/plotting
            cam_np = cam_upsampled.squeeze().cpu().numpy()
            pred_mask_vis = torch.argmax(pred_mask_upsampled, dim=0).cpu().numpy()
            gt_mask_np = mask.squeeze().cpu().numpy() # Ground truth mask (already on CPU)

            # Colorize mask function (simple version)
            def colorize_mask(mask, num_classes=3):
                # Define consistent colors: BG=Black, Pet=Red, Border=Green (adjust if needed)
                colors = np.array([[0,0,0], [255,0,0], [0,255,0]], dtype=np.uint8)
                # Handle cases where mask might have values outside expected range
                mask = np.clip(mask, 0, num_classes - 1)
                colored = colors[mask]
                return colored

            plt.figure(figsize=(16, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(img_np)
            plt.imshow(cam_np, cmap="jet", alpha=0.5)
            plt.title(f"CAM (Class {label})")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(img_np)
            plt.imshow(colorize_mask(pred_mask_vis, num_classes=model.class_head.out_features), alpha=0.5)
            plt.title("Predicted Segmentation")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(img_np)
            plt.imshow(colorize_mask(gt_mask_np, num_classes=model.class_head.out_features), alpha=0.5)
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Train Weakly-Supervised ResNet+SEC Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--sec-alpha', type=float, default=1.0, help='Weight for SEC seed loss')
    parser.add_argument('--sec-beta', type=float, default=1.0, help='Weight for SEC expand loss')
    parser.add_argument('--sec-gamma', type=float, default=0.5, help='Weight for SEC constrain loss')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes (Background, Pet, Border)')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    # --- Datasets ---
    print("Loading datasets...")
    # Training Dataset: Uses only image-level labels for weak supervision
    train_dataset_weak = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="binary-category", # Image-level labels {0: cat, 1: dog}
        transform=ResNetTransform, # Apply ResNet preprocessing
        download=True,
    )
    # Test Dataset: Uses segmentation masks for visualization
    # Use MultiTargetOxfordPet as it provides segmentation masks by default
    # And applies the necessary transforms including ResNetTransform for image
    test_dataset_seg = MultiTargetOxfordPet() # Provides (img, seg_mask)

    # Apply the 80/20 split using the fixed seed from utils.dataset
    # Note: We need to split indices, then create subsets for both datasets
    # Ensure TrainTestSplit uses a fixed seed internally (it does now)
    num_total = len(train_dataset_weak) # Should be same as len(test_dataset_seg)
    train_indices, test_indices = TrainTestSplit(range(num_total), 0.8)

    train_subset_weak = torch.utils.data.Subset(train_dataset_weak, train_indices.indices)
    test_subset_seg = torch.utils.data.Subset(test_dataset_seg, test_indices.indices)

    # --- Dataloaders ---
    train_loader_weak = DataLoader(
        train_subset_weak,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    # Visualization uses the test subset directly, no loader needed for that part

    print(f"Datasets loaded: {len(train_subset_weak)} train (weak labels), {len(test_subset_seg)} test/vis (seg masks).")

    # Train the weakly supervised model
    trained_model_path = train_weakly_supervised(
        train_loader_weak,
        args=args, # Pass the parsed arguments
        device=DEVICE,
        num_classes=args.num_classes
    )

    # Load the trained model
    print("Loading trained model for visualization...")
    model = MultiHeadResNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=DEVICE))
    model.to(DEVICE) # Ensure model is on the device

    # Visualize some predictions on the test set
    # Pass the test dataset subset and the device
    visualize_prediction(model, test_subset_seg, num_samples=5, device=DEVICE)

    print("\nTraining and visualization complete.")
    print(f"To evaluate this model, run: python -m src.evaluate --model-type resnet_sec --model-path {trained_model_path}")
