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

import matplotlib.pyplot as plt
from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.utils.dataset import TrainTestSplit
from src.utils.loss import SECLoss, DiceLoss, apply_dense_crf, denorm_image
from src.weakly_supervised.wss_deeplabv3 import WSSDeepLabV3
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)


def show_prediction(img_tensor, mask_pred, act_mask, output, cam):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()
    cam = cam.cpu().numpy()

    dice_loss = DiceLoss()
    loss = dice_loss(output, act_mask)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img)
    plt.imshow(cam, alpha=0.5)
    plt.title(f"CAM {1 - loss.item():.2f}")
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

    classifier = resnet18(pretrained=True).cuda()
    cam_extractor = GradCAMpp(classifier, "layer4")

    model = WSSDeepLabV3(num_classes=3)
    model = model.cuda()

    # Weakly supervised loss
    sec = SECLoss()
    dice = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, _, labels in bar:
            imgs = imgs.cuda()

            optimizer.zero_grad()
            pred_mask = model(imgs)
            probs = torch.softmax(pred_mask, dim=1)

            classifier_output = classifier(imgs)

            # Extract CAM with GradCAM++

            class_idx = torch.argmax(classifier_output, 1)
            activation_map = cam_extractor(
                list(class_idx.cpu().numpy()), classifier_output, retain_graph=True
            )
            c = F.interpolate(
                activation_map[0].unsqueeze(0),
                (imgs.size(2), imgs.size(3)),
                mode="bilinear",
            ).squeeze()

            cam = torch.zeros(
                (
                    pred_mask.size(0),
                    # pred_mask.size(1),
                    imgs.size(2),
                    imgs.size(3),
                ),
                dtype=torch.long,
            ).cuda()

            for batch_id in range(pred_mask.size(0)):
                # cam[batch_id, :, :] = (c[batch_id] > 0.5) * labels[batch_id]

                crf_input = np.zeros((3, imgs.size(2), imgs.size(3)))
                crf_input[labels[batch_id]] = c[batch_id].detach().cpu().numpy()

                background = 1.0 - np.max(crf_input, axis=0, keepdims=True)
                background = np.clip(background, 0, 1)

                crf_input[0] = background

                eps = 1e-8
                crf_input = crf_input / (np.sum(crf_input, axis=0, keepdims=True) + eps)

                crf_output = apply_dense_crf(denorm_image(imgs[batch_id]), crf_input)
                crf_output = torch.tensor(crf_output).to(pred_mask.device)

                cam[batch_id, :, :] = torch.argmax(crf_output, 0)

            loss = dice(probs, cam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), out_name)
    return model


def TestModel(model, train_set):
    model.eval()
    model.cuda()

    classifier = resnet18(pretrained=True).cuda()
    cam_extractor = GradCAMpp(classifier, "layer4")

    for i in range(50):
        idx = np.random.randint(len(train_set))
        img, mask, _ = train_set[idx]
        img = img.cuda()
        mask = mask.cuda()
        output = model(img.unsqueeze(0))
        pred = output.argmax(1).cpu().squeeze(0)

        # Extract CAM with GradCAM++
        classifier_output = classifier(img.unsqueeze(0))
        class_idx = torch.argmax(classifier_output, 1)
        activation_map = cam_extractor(
            list(class_idx.cpu().numpy()), classifier_output, retain_graph=True
        )
        cam = F.interpolate(
            activation_map[0].unsqueeze(0),
            (img.size(1), img.size(2)),
            mode="bilinear",
        ).squeeze()

        show_prediction(img, pred, mask, output, cam)


def LoadModel(model_path):
    model = WSSDeepLabV3(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def LoadClassifier(model_path):
    model = resnet18(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def Main():

    model_path = "./models/wss_deep_lab_v3_3_classes.pth"
    # model = LoadModel(model_path)
    model = TrainModel(num_epochs=5, out_name=model_path)

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)


if __name__ == "__main__":
    Main()
