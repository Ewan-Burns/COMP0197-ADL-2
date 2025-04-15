import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt

from src.MultiTargetOxfordPet import OxfordIIITPet


# Load and preprocess your image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


dataset = OxfordIIITPet(root="./data", split="trainval", target_types="binary-category", transform=transform, download=True)
input_tensor = dataset[0][0].unsqueeze(0)  # shape: [1, 3, 224, 224]

# Load model
model = resnet18(pretrained=True)
model.eval()

# Set up Grad-CAM++
cam_extractor = GradCAMpp(model, target_layer="layer4")  # pick a layer like "layer4"

# Forward pass
output = model(input_tensor)

# Generate CAM for the top-1 predicted class
class_idx = output.argmax().item()
activation_map = cam_extractor(class_idx, output)

# Overlay CAM on image
result = overlay_mask(img, activation_map[0], alpha=0.5)

# Show result
plt.imshow(result)
plt.axis('off')
plt.title(f"Grad-CAM++ for class {class_idx}")
plt.show()