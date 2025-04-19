import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models.segmentation as models
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.utils.dataset import TrainTestSplit
from src.utils.dice_loss import DiceLoss


def show_prediction(img_tensor, mask_pred, act_mask, output):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()

    dice_loss = DiceLoss()
    loss = dice_loss(output, act_mask)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Predicted Segmentation {1 - loss.item():.2f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(target, alpha=0.5)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()


# Update signature to accept pretrained flag
def CreateDeepLabV3(num_classes, pretrained=True):
    """Creates a DeepLabV3 model with a ResNet50 backbone."""
    model = models.deeplabv3_resnet50(weights='DEFAULT' if pretrained else None)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# Update function signature to accept device, loaders, lr etc.
def TrainModel(train_loader, test_loader, device, num_epochs=5, lr=1e-4, loss_balance=np.array([0.5, 0.5]), out_name="", num_classes=3):
    """Trains the DeepLabV3 model."""
    print(f"Starting DeepLabV3 Training on {device}...")
    model = CreateDeepLabV3(num_classes=num_classes, pretrained=True)
    model = model.to(device) # Move model to device

    criterion = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss() # Instantiate DiceLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    alpha, beta = loss_balance / loss_balance.sum()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, masks in bar:
            imgs = imgs.to(device) # Move data to device
            masks = masks.squeeze(1).long().to(device) # Move data to device

            optimizer.zero_grad()
            output = model(imgs)["out"]

            d = dice_loss_fn(output, masks) # Use instantiated loss
            ce = criterion(output, masks)
            loss = alpha * ce + beta * d

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item(), dice=d.item(), ce=ce.item())

    torch.save(model.state_dict(), out_name)
    print(f"Model saved to {out_name}")

    # --- Optional: Validation Loss Calculation ---
    # print("Calculating validation loss...")
    # model.eval()
    # val_loss = 0.0
    # val_dice = 0.0
    # val_ce = 0.0
    # with torch.no_grad():
    #     val_bar = tqdm(test_loader, desc="Validation", leave=False)
    #     for imgs, masks in val_bar:
    #         imgs = imgs.to(device)
    #         masks = masks.squeeze(1).long().to(device)
    #         output = model(imgs)["out"]
    #         d = dice_loss_fn(output, masks)
    #         ce = criterion(output, masks)
    #         loss = alpha * ce + beta * d
    #         val_loss += loss.item()
    #         val_dice += d.item()
    #         val_ce += ce.item()
    #         val_bar.set_postfix(loss=loss.item(), dice=d.item(), ce=ce.item())
    # avg_val_loss = val_loss / len(test_loader)
    # avg_val_dice = val_dice / len(test_loader)
    # avg_val_ce = val_ce / len(test_loader)
    # print(f"Avg Validation Loss: {avg_val_loss:.4f} (Dice: {avg_val_dice:.4f}, CE: {avg_val_ce:.4f})")
    # --- End Optional Validation ---

    return out_name # Return path instead of model


# Update function signature to accept device and dataset
def TestModel(model, dataset, device, num_samples=5):
    """Visualizes predictions for a few samples from the dataset."""
    print(f"Visualizing predictions on {device}...")
    model.eval()
    model.to(device) # Ensure model is on the correct device
    with torch.no_grad():
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        for i in indices:
            img, mask = dataset[i] # Get image and mask from dataset
            img_input = img.unsqueeze(0).to(device) # Move input to device
            mask_input = mask.long().to(device) # Move mask to device (needed for show_prediction loss calc)

            output = model(img_input)["out"] # Get model output
            pred = output.argmax(1).cpu().squeeze(0) # Get prediction and move to CPU

            # Pass original image tensor (before moving to device) and CPU prediction/mask
            show_prediction(img, pred, mask.squeeze(0).cpu(), output.cpu())


# Update signature to accept device and num_classes
def LoadModel(model_path, num_classes=3, device='cpu'):
    """Loads a trained DeepLabV3 model."""
    # Use pretrained=False when loading state dict
    model = CreateDeepLabV3(num_classes=num_classes, pretrained=False)
    # Use map_location to load onto the correct device
    # Set strict=False to ignore unexpected keys like aux_classifier
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device) # Ensure model is on the device
    return model


if __name__ == "__main__":
    # Determine device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Define other parameters
    BATCH_SIZE = 16 # Adjust based on memory
    NUM_WORKERS = 4
    NUM_EPOCHS = 5 # Adjust as needed
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 3
    MODEL_PATH = "./models/deep_lab_v3_3_classes.pth"

    # Dataset and Dataloaders
    print("Loading dataset...")
    full_dataset = MultiTargetOxfordPet()
    # Use fixed 80/20 split for consistency if evaluating later
    # generator = torch.Generator().manual_seed(42) # Use fixed seed if splitting needed
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2], generator=generator)
    # For this script, train on the full dataset
    train_dataset = full_dataset
    test_dataset = full_dataset # Use full dataset for validation loss check / visualization

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    # Create a loader for the test/validation set if needed for validation loss or separate visualization
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    print(f"Dataset loaded: {len(train_dataset)} samples.")

    # Train the model
    trained_model_path = TrainModel(
        train_loader=train_loader,
        test_loader=test_loader, # Pass test_loader for optional validation
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        out_name=MODEL_PATH,
        num_classes=NUM_CLASSES
    )

    # Load the trained model
    print("Loading trained model for visualization...")
    model = LoadModel(trained_model_path, num_classes=NUM_CLASSES, device=DEVICE)

    # Visualize predictions using the test_dataset
    TestModel(model, test_dataset, device=DEVICE, num_samples=5)

    print("\nTraining and visualization complete.")
    print(f"To evaluate this model, run: python -m src.evaluate --model-type deeplabv3 --model-path {trained_model_path}")
