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
from src.utils.loss import DiceLoss
from src.utils.device import get_device
from src.utils.random_utils import set_seed, worker_init_fn

# Set fixed random seed
SEED = 42
set_seed(SEED)

device = get_device()

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


def CreateDeepLabV3(num_classes):
    torch.manual_seed(SEED)
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def TrainModel(num_epochs=5, loss_balance=np.array([0.5, 0.5]), out_name=""):
    dataset = MultiTargetOxfordPet()
    train_set, test_set = TrainTestSplit(dataset, 0.8)

    # Create a deterministic generator for shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_set, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    model = CreateDeepLabV3(num_classes=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    alpha, beta = loss_balance / loss_balance.sum()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, masks, _ in bar:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).to(device).long()  # Ensure masks are of type LongTensor

            optimizer.zero_grad()
            output = model(imgs)["out"]

            d = dice_loss(output, masks)
            ce = criterion(output, masks)  # No more type mismatch
            loss = alpha * ce + beta * d

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item(), dice=d.item(), ce=ce.item())

    torch.save(model.state_dict(), out_name)

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_dice_loss = 0.0
        running_ce_loss = 0.0
        bar = tqdm(test_loader, desc=f"Validation", leave=True)

        for imgs, masks in bar:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long.to(device)

            optimizer.zero_grad()
            output = model(imgs)["out"]

            d = dice_loss(output, masks)
            ce = criterion(output, masks)
            loss = alpha * ce + beta * d

            running_dice_loss += d.item()
            running_ce_loss += ce.item()
            running_loss += loss.item()
            bar.set_postfix(loss=loss.item(), dice=d.item(), ce=ce.item())

        print(f"Validation loss: {running_loss}")
        print(f"Dice Loss: {running_dice_loss}")
        print(f"Cross-Entropy Loss: {running_ce_loss}")

    return model


def TestModel(model, train_set):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in range(50):
            idx = np.random.randint(len(train_set))
            img, mask = train_set[idx]
            img = img.to(device)
            mask = mask.long.to(device)
            output = model(img.unsqueeze(0))["out"]
            pred = output.argmax(1).cpu().squeeze(0)
            show_prediction(img, pred, mask, output)


def LoadModel(model_path):
    model = CreateDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    # Define model path
    model_path = "./models/deep_lab_v3_3_classes.pth"
    
    # Uncomment to train
    model = TrainModel(num_epochs=5, out_name=model_path)
    
    # Or uncomment to load existing model
    # model = LoadModel(model_path)

    # Test model visualization
    dataset = MultiTargetOxfordPet()
    _, test_set = TrainTestSplit(dataset, 0.8)
    TestModel(model, test_set)


if __name__ == "__main__":
    main()
