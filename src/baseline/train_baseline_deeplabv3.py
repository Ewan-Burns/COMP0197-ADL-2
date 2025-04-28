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
from src.utils.viz import show_prediction


def CreateDeepLabV3(num_classes):
    model = models.deeplabv3_resnet50(pretrained_backbone=True, num_classes=num_classes)
    return model


def TrainModel(num_epochs=5, loss_balance=np.array([0.5, 0.5]), out_name=""):
    dataset = MultiTargetOxfordPet()
    train_set, test_set = TrainTestSplit(dataset, 0.8)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(train_set, batch_size=16, num_workers=8)

    model = CreateDeepLabV3(num_classes=3)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    alpha, beta = loss_balance / loss_balance.sum()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, masks, _ in bar:
            imgs = imgs.cuda()
            masks = masks.squeeze(1).cuda()

            optimizer.zero_grad()
            output = model(imgs)["out"]

            d = dice_loss(output, masks)
            ce = criterion(output, masks)
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

        for imgs, masks, _ in bar:
            imgs = imgs.cuda()
            masks = masks.squeeze(1).cuda()

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
    model.cuda()
    with torch.no_grad():
        for i in range(50):
            idx = np.random.randint(len(train_set))
            img, mask, _ = train_set[idx]
            img = img.cuda()
            mask = mask.cuda()
            output = model(img.unsqueeze(0))["out"]
            pred = output.argmax(1).cpu().squeeze(0)
            show_prediction(img, pred, mask, output)


def LoadModel(model_path):
    model = CreateDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def Main():

    model_path = "./models/baseline_deep_lab_v3_3_classes.pth"
    # model = LoadModel(model_path)
    model = TrainModel(num_epochs=5, out_name=model_path)

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    Main()
