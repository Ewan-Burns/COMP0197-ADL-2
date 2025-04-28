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

from src.weakly_supervised.resnet import MultiHeadResNet, generate_cams
from src.utils.sec import sec_loss
from src.utils.dataset import TrainTestSplit, ResNetTransform
from src.utils.device import get_device
from src.utils.random_utils import set_seed, worker_init_fn
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# Set fixed random seed
SEED = 42
set_seed(SEED)

device = get_device()

if __name__ == "__main__":
    dataset = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="binary-category",
        transform=ResNetTransform,
        download=True,
    )
    dataset, _ = TrainTestSplit(dataset, 0.05)
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

    model = MultiHeadResNet(num_classes=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(num_epochs):

        bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for imgs, labels in bar:
            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_onehot = (
                torch.nn.functional.one_hot(labels, num_classes=3).float().to(device)
            )

            feats, segm, cl = model(imgs)
            probs = torch.softmax(segm, dim=1)
            probs = F.interpolate(
                probs, size=(224, 224), mode="bilinear", align_corners=False
            )

            # CAM seeds
            fc_weights = model.class_head.weight.detach()
            class_ids = labels.to(device)

            cams = generate_cams(feats, fc_weights, class_ids)

            # Resize CAMs to match seg_head output
            seed_masks = torch.nn.functional.interpolate(
                cams, size=probs.shape[2:], mode="bilinear", align_corners=False
            )

            # Convert PIL imgs to numpy for CRF
            imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy() * 255
            imgs_np = imgs_np.astype("uint8")

            loss = sec_loss(probs, seed_masks, imgs_np, labels)
            loss += criterion(cl, labels)

            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def visualize_prediction(img, cam, pred_probs):
        """
        img: Tensor (3, H, W)
        cam: Tensor (1, H, W)
        pred_probs: Tensor (C, H, W)
        """

        # Convert to displayable formats
        img_np = img.permute(1, 2, 0).cpu().numpy()
        cam_np = cam.squeeze().cpu().numpy()
        pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy()

        # Optional: color mapping for mask
        def colorize_mask(mask):
            colors = np.random.randint(0, 255, size=(pred_probs.shape[0], 3))
            colored = colors[mask]
            return colored.reshape(mask.shape[0], mask.shape[1], 3)

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img_np)
        plt.title("Input Image")

        plt.subplot(1, 4, 2)
        plt.imshow(img_np)
        plt.imshow(cam_np, cmap="jet", alpha=0.5)
        plt.title("Class Activation Map (CAM)")

        plt.subplot(1, 4, 3)
        plt.imshow(img_np)
        plt.imshow(cam_np > 0.5, alpha=0.5)
        plt.title("Class Activation Map (CAM)")

        plt.subplot(1, 4, 4)
        plt.imshow(img_np)
        plt.imshow(colorize_mask(pred_mask), alpha=0.5)
        plt.title("Predicted Segmentation")

        plt.tight_layout()
        plt.show()

    # Evaluation mode
    model.eval()

    # Load dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Show predictions for first 5 images
    with torch.no_grad():
        for i in range(5):
            img, label = dataset[i]
            img_input = img.unsqueeze(0).to(device)

            # Get features and logits
            feats, segm, cl = model(img_input)
            pred_mask = F.softmax(segm, dim=1)

            # Generate CAM
            fc_weights = model.class_head.weight.detach()
            class_idx = label
            cam = torch.einsum(
                "chw,c->hw", feats[0], fc_weights[class_idx]
            )  # Single CAM
            cam = torch.relu(cam)
            cam -= cam.min()
            cam /= cam.max() + 1e-6
            cam = cam.unsqueeze(0)  # (1, H, W)

            # Upsample to input size
            cam = F.interpolate(
                cam.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            )[0]
            pred_mask = F.interpolate(
                pred_mask, size=(224, 224), mode="bilinear", align_corners=False
            )[0]

            visualize_prediction(img, cam, pred_mask)
