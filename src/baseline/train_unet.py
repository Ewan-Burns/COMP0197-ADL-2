import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from .unet import UNet # Correct relative import to specify the module file
from src.utils.dice_loss import DiceLoss # Import the DiceLoss CLASS
from tqdm import tqdm
from src.MultiTargetOxfordPet import MultiTargetOxfordPet



def show_prediction(img_tensor, mask_pred, act_mask):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(target)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()


# Update function signature to accept device
def TrainModel(train_loader, device, num_epochs=5, lr=1e-4):
    """Trains the U-Net model."""
    print(f"Starting U-Net Training on {device}...")
    model = UNet(n_classes=3).to(device) # Move model to device
    criterion = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss() # Instantiate the DiceLoss class
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs): # Use num_epochs from args
        model.train()
        epoch_loss = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for images, masks in bar:
            images = images.to(device) # Move data to device
            masks = masks.squeeze(1).long().to(device) # Move data to device

            outputs = model(images)
            ce_loss = criterion(outputs, masks)
            # Call the instantiated DiceLoss object
            dice = dice_loss_fn(outputs, masks)
            loss = ce_loss + dice # Combine CE loss and Dice loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    model_path = "./models/unet_3_classes.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Return model path for consistency, though not strictly needed here
    return model_path


# Update function signature to accept device
def TestModel(model, train_set, device):
    """Visualizes predictions for a few samples."""
    print(f"Visualizing predictions on {device}...")
    model.to(device) # Ensure model is on the correct device
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(train_set), 5, replace=False) # Show random samples
        for i in indices:
            sample_img, act_mask = train_set[i]
            img = sample_img.unsqueeze(0).to(device) # Move input to device
            output = model(img)
            pred_mask = output.argmax(1).squeeze(0).cpu() # Move prediction to CPU for plotting
            show_prediction(sample_img, pred_mask, act_mask)

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

    # Dataset and Dataloader
    print("Loading dataset...")
    # Use the full dataset for training in this simplified script
    train_set = MultiTargetOxfordPet()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Dataset loaded: {len(train_set)} samples.")

    # Train the model
    # Pass device and other params to TrainModel
    trained_model_path = TrainModel(train_loader, device=DEVICE, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # Load the trained model
    print("Loading trained model for visualization...")
    model = UNet(n_classes=NUM_CLASSES)
    # Use map_location to load onto the correct device
    model.load_state_dict(torch.load(trained_model_path, map_location=DEVICE))
    model.to(DEVICE) # Ensure model is on the device after loading

    # Visualize predictions
    # Pass device to TestModel
    TestModel(model, train_set, device=DEVICE)

    print("\nTraining and visualization complete.")
    print(f"To evaluate this model, run: python -m src.evaluate --model-type unet --model-path {trained_model_path}")
