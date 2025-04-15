import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

from src.MultiTargetOxfordPet import OxfordIIITPet, MultiTargetOxfordPet


# Apply CRF
def apply_crf_to_heatmap(heatmap, input_image, num_classes=2):
    # Convert the heatmap to 3D (with channels for each class)
    probs = np.stack([1 - heatmap, heatmap])
    unary = unary_from_softmax(probs)

    # Apply CRF using the unary potentials (heatmap) and the input image (for pairwise potentials)
    crf = dcrf.DenseCRF2D(input_image.shape[1], input_image.shape[0], num_classes)
    crf.setUnaryEnergy(unary)

    # Pairwise potentials for spatial smoothness (using Gaussian pairwise potential)
    image_2d = np.ascontiguousarray(input_image)
    crf.addPairwiseGaussian(sxy=3, compat=10)
    crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_2d, compat=10)

    # Run CRF inference
    refined_mask = crf.inference(5)

    # Convert the result to a probability map (if needed)
    refined_mask = np.argmax(np.array(refined_mask).reshape((num_classes, -1)), axis=0)
    return refined_mask.reshape(input_image.shape[0], input_image.shape[1])


# Load and preprocess your image
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = MultiTargetOxfordPet()

# Load model
model = resnet18(pretrained=True)
model.eval()


# Set up Grad-CAM++
cam_extractor = GradCAMpp(model, target_layer="layer4")  # pick a layer like "layer4"

for _ in range(10):
    img, masks = dataset[np.random.randint(1000)]
    input_tensor = img.unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Forward pass
    output = model(input_tensor)

    # Generate CAM for the top-1 predicted class

    class_idx = output.argmax().item()
    classes = sorted(enumerate(output.detach().numpy().ravel()), key=lambda s: s[1])[
        -4:
    ]
    classes = [id for id, _ in classes]

    activation_map = cam_extractor(class_idx, output)

    # Overlay CAM on image
    cam = (
        F.interpolate(
            activation_map[0].unsqueeze(0), (img.size(1), img.size(2)), mode="bilinear"
        )
        .squeeze()
        .squeeze()
        .numpy()
    )
    cam_thr = cam > 0.5

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denormalized_image = img * std[:, None, None] + mean[:, None, None]
    denormalized_image = (
        denormalized_image * 255
    )  # Convert back to pixel range [0, 255]
    denormalized_image = (
        denormalized_image.permute(1, 2, 0).byte().cpu().numpy()
    )  # Convert to numpy for CRF
    crf = apply_crf_to_heatmap(cam, denormalized_image)

    # Show result
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(denormalized_image)
    plt.imshow(cam_thr, alpha=0.5)
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
