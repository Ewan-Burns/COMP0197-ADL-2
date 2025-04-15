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
from src.utils.dice_loss import dice_loss

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
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(target, alpha=0.5)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()

def CreateDeepLabV3(num_classes):
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def TrainModel(
        num_epochs = 5,
        loss_balance = np.array([0.5, 0.5]),
        out_name = ""
):
    train_set = MultiTargetOxfordPet()
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    model = CreateDeepLabV3(num_classes=3)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    alpha, beta = loss_balance / loss_balance.sum()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, masks in bar:
            imgs = imgs.cuda()
            masks = masks.squeeze(1).cuda()

            optimizer.zero_grad()
            output = model(imgs)["out"]
            loss = alpha * criterion(output, masks) + beta * dice_loss(
                F.softmax(output, dim=1).float(),
                F.one_hot(masks, 3).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), out_name)

def TestModel(model, train_set):
    model.eval()
    model.cuda()
    with torch.no_grad():

        for i in range(50):
            idx = np.random.randint(len(train_set))
            img, mask = train_set[idx]
            img = img.cuda()
            output = model(img.unsqueeze(0))["out"].argmax(1).cpu()
            pred = output.squeeze(0)
            show_prediction(img, pred, mask)

if __name__ == "__main__":

    model_path = "./models/deep_lab_v3_3_classes.pth"
    #TrainModel(num_epochs=5, out_name=model_path)

    model = CreateDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)
