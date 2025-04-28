import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchcam.methods import GradCAMpp
from torchvision.models import resnet18

from src.weakly_supervised.resnet import MultiHeadResNet
from src.utils.dataset import TrainTestSplit, ResNetTransform
from src.utils.loss import DiceLoss, apply_dense_crf, denorm_image, batched_cam_to_crf
from src.utils.viz import show_prediction
from src.MultiTargetOxfordPet import MultiTargetOxfordPet


def TrainModel(num_epochs, out_name):
    dataset = MultiTargetOxfordPet()
    dataset, _ = TrainTestSplit(dataset, 0.8)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=6)

    model = MultiHeadResNet(num_classes=3).cuda()
    classifier = resnet18(pretrained=True).cuda()
    cam_extractor = GradCAMpp(classifier, "layer4")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice = DiceLoss()

    for epoch in range(num_epochs):
        bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for imgs, labels, _ in bar:
            optimizer.zero_grad()

            imgs = imgs.cuda()
            segm = model(imgs)
            probs = torch.softmax(segm, dim=1)
            probs = F.interpolate(
                probs, size=(224, 224), mode="bilinear", align_corners=False
            )

            classifier_output = classifier(imgs)

            # Extract CAM with GradCAM++

            class_idx = torch.argmax(classifier_output, 1)
            activation_map = cam_extractor(
                list(class_idx.cpu().numpy()),
                classifier_output,
            )
            cam = F.interpolate(
                activation_map[0].unsqueeze(0),
                (imgs.size(2), imgs.size(3)),
                mode="bilinear",
            ).squeeze()

            crf = batched_cam_to_crf(cam, imgs, labels)
            loss = dice(probs, crf)

            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), out_name)
    return model


def TestModel(model, train_set):
    model.eval()
    model.cuda()

    classifier = resnet18(pretrained=True).cuda()
    cam_extractor = GradCAMpp(classifier, "layer4")

    for i in range(50):
        idx = np.random.randint(len(train_set))
        img, mask, label = train_set[idx]
        img = img.cuda()
        mask = mask.cuda()
        output = model(img.unsqueeze(0))
        output = F.interpolate(
            output, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Extract CAM with GradCAM++
        classifier_output = classifier(img.unsqueeze(0))
        class_idx = torch.argmax(classifier_output, 1)
        activation_map = cam_extractor(
            list(class_idx.cpu().numpy()),
            classifier_output,
        )

        cam = F.interpolate(
            activation_map[0].unsqueeze(0),
            (img.size(2), img.size(3)),
            mode="bilinear",
        )

        crf = batched_cam_to_crf(cam, img.unsqueeze(0), [label])
        pred = output.argmax(1).cpu().squeeze(0)
        show_prediction(img, pred, mask, output, crf)


def LoadModel(model_path):
    model = MultiHeadResNet(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def Main():
    model_path = "./models/wss_resnet_3_classes.pth"
    model = LoadModel(model_path)
    # model = TrainModel(num_epochs=1, out_name=model_path)

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    Main()
