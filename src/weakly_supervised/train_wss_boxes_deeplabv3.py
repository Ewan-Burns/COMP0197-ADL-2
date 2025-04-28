import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models.segmentation as models
from torchcam.methods import GradCAMpp
from torchvision.models import resnet18
from tqdm import tqdm

from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.utils.dataset import TrainTestSplit
from src.utils.loss import (
    SECLoss,
    DiceLoss,
    apply_dense_crf,
    denorm_image,
    batched_cam_to_crf,
)
from src.utils.viz import show_prediction
from src.weakly_supervised.wss_deeplabv3 import WSSDeepLabV3


def TrainClassifier(train_loader, num_epochs=5, out_name=""):
    model = resnet18(num_classes=3)
    model = model.cuda()

    # Classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, _, labels in bar:
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), out_name)
    return model


def TrainModel(num_epochs=5, out_name=""):
    dataset = MultiTargetOxfordPet()
    train_set, test_set = TrainTestSplit(dataset, 0.8)

    train_loader = DataLoader(train_set, batch_size=6, shuffle=True, num_workers=6)
    test_loader = DataLoader(train_set, batch_size=6, num_workers=6)

    model = WSSDeepLabV3(num_classes=3)
    model = model.cuda()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, _, box, labels in bar:
            imgs = imgs.cuda()

            optimizer.zero_grad()
            pred_mask = model(imgs)
            probs = torch.softmax(pred_mask, dim=1)

            # Hard label box

            boxes = torch.zeros((probs.size(0), 1, probs.size(2), probs.size(3)))
            for batch_id in range(probs.size(0)):
                boxes[
                    batch_id,
                    0,
                    box[batch_id][0][1]
                    .long()
                    .item() : box[batch_id][0][3]
                    .long()
                    .item(),
                    box[batch_id][0][0]
                    .long()
                    .item() : box[batch_id][0][2]
                    .long()
                    .item(),
                ] = 1

            crf = batched_cam_to_crf(boxes, imgs, labels)
            loss = dice(probs, crf)
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
        img, mask, box, label = train_set[idx]
        img = img.cuda()
        mask = mask.cuda()
        output = model(img.unsqueeze(0))
        pred = output.argmax(1).cpu().squeeze(0)

        ymin = box[0][0].long().item()
        xmin = box[0][1].long().item()
        ymax = box[0][2].long().item()
        xmax = box[0][3].long().item()

        # Extract CAM with GradCAM++
        boxes = torch.zeros(output.size(2), output.size(3)).cuda()
        boxes[xmin:xmax, ymin:ymax] = 1

        crf = batched_cam_to_crf(boxes.unsqueeze(0), img.unsqueeze(0), [label])
        show_prediction(img, pred, mask, output, crf)


def LoadModel(model_path):
    model = WSSDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def LoadClassifier(model_path):
    model = resnet18(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def Main():

    model_path = "./models/wss_boxes_deep_lab_v3_3_classes.pth"
    model = LoadModel(model_path)
    # model = TrainModel(num_epochs=1, out_name=model_path)

    train_set = MultiTargetOxfordPet(random_vflip=False, random_hflip=False)
    TestModel(model, train_set)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    Main()
