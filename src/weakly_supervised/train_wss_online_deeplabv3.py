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
from src.utils.loss import SECLoss, DiceLoss
from src.weakly_supervised.wss_online_deeplabv3 import OnlineWSSDeepLabV3


def show_prediction(img_tensor, mask_pred, act_mask, output, cam):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()
    cam = cam.argmax(1).squeeze().cpu().numpy()

    dice_loss = DiceLoss()
    loss = dice_loss(output, act_mask)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img)
    plt.imshow(cam, alpha=0.5)
    plt.title(f"Predicted Segmentation {1 - loss.item():.2f}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Predicted Segmentation {1 - loss.item():.2f}")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.imshow(target, alpha=0.5)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()


def TrainModel(num_epochs=5, out_name=""):
    dataset = MultiTargetOxfordPet()
    train_set, test_set = TrainTestSplit(dataset, 0.8)

    train_loader = DataLoader(train_set, batch_size=6, shuffle=True, num_workers=6)
    test_loader = DataLoader(train_set, batch_size=6, num_workers=6)

    model = OnlineWSSDeepLabV3(num_classes=3)
    model = model.cuda()

    # Weakly supervised loss
    sec = SECLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, _, labels in bar:
            imgs = imgs.cuda()

            optimizer.zero_grad()
            pred_mask, cam = model(imgs)
            probs = torch.softmax(pred_mask, dim=1)

            labels_onehot = (
                torch.nn.functional.one_hot(labels, num_classes=3).float().cuda()
            )

            loss = sec(probs, cam, imgs, labels_onehot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), out_name)
    return model


def TestModel(model, train_set):
    model.eval()
    model.cuda()

    for i in range(50):
        idx = np.random.randint(len(train_set))
        img, mask, _ = train_set[idx]
        img = img.cuda()
        mask = mask.cuda()
        output, cam = model(img.unsqueeze(0))
        pred = output.argmax(1).cpu().squeeze(0)
        show_prediction(img, pred, mask, output, cam)


def LoadModel(model_path):
    model = OnlineWSSDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def Main():

    model_path = "./models/wss_online_deep_lab_v3_3_classes.pth"
    # model = LoadModel(model_path)
    model = TrainModel(num_epochs=3, out_name=model_path)

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)


if __name__ == "__main__":
    Main()
