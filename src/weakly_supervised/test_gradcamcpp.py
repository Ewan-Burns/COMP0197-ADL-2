import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.MultiTargetOxfordPet import OxfordIIITPet, MultiTargetOxfordPet


def Main():
    dataset = MultiTargetOxfordPet()

    # Load model
    model = resnet18(pretrained=True)
    model.eval()

    # Set up Grad-CAM++
    cam_extractor = GradCAMpp(
        model, target_layer="layer4"
    )  # pick a layer like "layer4"

    for _ in range(10):
        img, masks, _ = dataset[np.random.randint(1000)]
        input_tensor = img.unsqueeze(0)  # shape: [1, 3, 224, 224]

        # Forward pass
        output = model(input_tensor)

        classes = sorted(
            enumerate(output.detach().numpy().ravel()), key=lambda s: s[1]
        )[-4:]
        classes = [id for id, _ in classes]

        cam = np.zeros((output.size(1), img.size(1), img.size(2)))

        # Generate CAM for each of the classes
        for class_idx in range(output.size(1)):
            retain_graph = class_idx != output.shape[1] - 1
            activation_map = cam_extractor(class_idx, output, retain_graph=retain_graph)

            # Overlay CAM on image
            cam[class_idx] = (
                F.interpolate(
                    activation_map[0].unsqueeze(0),
                    (img.size(1), img.size(2)),
                    mode="bilinear",
                )
                .squeeze()
                .squeeze()
                .numpy()
            )

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        denormalized_image = img * std[:, None, None] + mean[:, None, None]
        denormalized_image = (
            denormalized_image * 255
        )  # Convert back to pixel range [0, 255]
        denormalized_image = (
            denormalized_image.permute(1, 2, 0).byte().cpu().numpy()
        )  # Convert to numpy for CRF
        crf = apply_crf_to_heatmap(cam, denormalized_image, 1000)

        # Show result
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(denormalized_image)
        plt.imshow(cam.argmax(0), alpha=0.5)
        plt.axis("off")
        plt.title(f"Grad-CAM++ for class {classes}")

        plt.subplot(1, 3, 2)
        plt.imshow(denormalized_image)
        plt.imshow(crf, alpha=0.5)
        plt.axis("off")
        plt.title(f"CRF")

        plt.subplot(1, 3, 3)
        plt.imshow(denormalized_image)
        plt.imshow(masks.permute(1, 2, 0).numpy(), alpha=0.5)
        plt.axis("off")
        plt.title(f"Target segmentation")

        plt.show()


if __name__ == "__main__":
    Main()
