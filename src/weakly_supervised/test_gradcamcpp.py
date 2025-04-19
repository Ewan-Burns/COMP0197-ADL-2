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

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and move to device
model = resnet18(weights='DEFAULT') # Use recommended way to load pretrained weights
model = model.to(device)
model.eval()


# Set up Grad-CAM++
cam_extractor = GradCAMpp(model, target_layer="layer4")  # pick a layer like "layer4"

for _ in range(10):
    # Get image (already a tensor from dataset) and mask
    img, masks = dataset[np.random.randint(len(dataset))] # Use len(dataset)
    input_tensor = img.unsqueeze(0).to(device)  # Move input tensor to device

    # Forward pass
    with torch.no_grad(): # Use no_grad for inference
        output = model(input_tensor)

    # Generate CAM for the top-1 predicted class
    # Ensure output is on CPU for numpy operations if needed by torchcam internals or subsequent code
    output_cpu = output.cpu()
    class_idx = output_cpu.argmax().item()

    # Get top classes (optional, for title) - use output_cpu
    # classes_scores = sorted(enumerate(output_cpu.detach().numpy().ravel()), key=lambda s: s[1], reverse=True)
    # top_classes_str = ", ".join([f"{idx}" for idx, score in classes_scores[:3]]) # Example: top 3 classes

    # torchcam expects model output on the same device as the model
    activation_map = cam_extractor(class_idx, output) # Pass original output tensor (on device)

    # Overlay CAM on image
    cam = (
        F.interpolate(
            activation_map[0].unsqueeze(0), (img.size(1), img.size(2)), mode="bilinear"
        )
        .squeeze()
        .squeeze()
        .cpu() # Move CAM to CPU before numpy conversion
        .numpy()
    )
    cam_thr = cam > 0.5 # Thresholding on CPU numpy array

    # Denormalize the original image tensor (img is on CPU)
    mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_cpu = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormalized_image_tensor = img * std_cpu + mean_cpu
    denormalized_image = (denormalized_image_tensor * 255).byte() # Convert to byte tensor [0, 255]
    # Permute and convert to numpy for CRF/plotting
    denormalized_image_np = denormalized_image.permute(1, 2, 0).numpy()

    # Apply CRF using the numpy image
    crf = apply_crf_to_heatmap(cam, denormalized_image_np)

    # Show result using the numpy image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(denormalized_image_np) # Show numpy image
    plt.imshow(cam_thr, alpha=0.5)
    plt.axis("off")
    plt.title(f"Grad-CAM++ Overlay (Class {class_idx})") # Use class_idx

    plt.subplot(1, 3, 2)
    plt.imshow(denormalized_image_np) # Show numpy image
    plt.imshow(crf, alpha=0.5)
    plt.axis("off")
    plt.title(f"CRF Refinement")

    plt.subplot(1, 3, 3)
    plt.imshow(denormalized_image_np) # Show numpy image
    # Ensure mask is numpy and correct shape for imshow
    mask_np = masks.squeeze().cpu().numpy() # Remove channel dim if present, move to CPU
    plt.imshow(mask_np, alpha=0.5)
    plt.axis("off")
    plt.title(f"Ground Truth Mask")

    plt.show()
