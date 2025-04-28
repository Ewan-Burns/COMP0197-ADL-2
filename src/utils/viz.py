import torch.nn.functional as F
from src.utils.loss import DiceLoss, denorm_image


import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


def show_prediction(img_tensor, mask_pred, act_mask, output, cam):
    img = denorm_image(img_tensor)

    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()
    cam_cpu = cam.squeeze().cpu().numpy()

    dice_loss = DiceLoss()
    loss = dice_loss(
        F.one_hot(mask_pred.unsqueeze(0).cuda(), 3).permute(0, 3, 1, 2).float()
        * 1000000,
        act_mask,
    )
    cam_loss = dice_loss(
        F.one_hot(cam, 3).permute(0, 3, 1, 2).float() * 1000000, act_mask
    )

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img)
    plt.imshow(cam_cpu, alpha=0.5)
    plt.title(f"GradCAM+CRF {1 - cam_loss.item():.2f} DICE")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Predicted Segmentation {1 - loss.item():.2f} DICE")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.imshow(target, alpha=0.5)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()
