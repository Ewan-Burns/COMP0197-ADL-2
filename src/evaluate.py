import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.utils.dataset import TrainTestSplit
from src.utils.metrics import calculate_iou, calculate_miou, calculate_dice, calculate_mdice
from src.baseline.train_baseline_deeplabv3 import CreateDeepLabV3 
from src.weakly_supervised.resnet import MultiHeadResNet
from src.utils.device import get_device

device = get_device()

def load_model(model_type, model_path, num_classes, device):
    """Loads the specified model architecture and state dict."""
    if model_type == 'deeplabv3':
        # use factory function
        model = CreateDeepLabV3(num_classes=num_classes, pretrained=False)
    elif model_type == 'resnet_sec':
        model = MultiHeadResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'deeplabv3', or 'resnet_sec'.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # set model to evaluation mode
    print(f"Loaded {model_type} model from {model_path}")
    return model

def evaluate(model, dataloader, device, num_classes):
    """Runs the evaluation loop and returns metrics."""
    print("Starting Evaluation...")
    model.eval() # ensure model is in eval mode
    all_ious = []
    all_dices = []
    with torch.no_grad():
        bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, masks in bar:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device) # N, H, W

            # handle different model output structures 
            if isinstance(model, MultiHeadResNet):
                 # UNet output: (N, C, H, W)
                 # MultiHeadResNet output: feats, segm, cl -> we need segm
                 _, outputs, _ = model(images) if isinstance(model, MultiHeadResNet) else (None, model(images), None)
                 # Upsample if needed (ResNet output might be smaller)
                 if outputs.shape[2:] != masks.shape[1:]:
                     outputs = torch.nn.functional.interpolate(
                         outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
                     )
            elif hasattr(model, 'classifier'): # heuristic for DeepLabV3
                 outputs = model(images)['out'] # N, C, H, W
            else:
                 raise TypeError(f"Model output handling not defined for type: {type(model)}")

            preds = torch.argmax(outputs, dim=1) # N, H, W

            # do all metric calculations
            for i in range(preds.shape[0]):
                iou = calculate_iou(preds[i], masks[i], num_classes)
                dice_score = calculate_dice(preds[i], masks[i], num_classes)
                all_ious.append(iou)
                all_dices.append(dice_score)

    miou = calculate_miou(all_ious)
    mdice = calculate_mdice(all_dices) # mDice is equivalent to mF1
    print(f"Evaluation Complete. mIoU: {miou:.4f}, mDice/mF1: {mdice:.4f}")
    return miou, mdice

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Segmentation Model')
    parser.add_argument('--model-type', type=str, required=True, choices=['deeplabv3', 'resnet_sec'],
                        help='Type of model architecture to load.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model state_dict (.pth file).')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu).')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    print("Loading test dataset...")
    full_dataset = MultiTargetOxfordPet() 
    test_indices = TrainTestSplit(range(len(full_dataset)), 0.8)[1].indices
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Test dataset loaded: {len(test_dataset)} samples.")

    model = load_model(args.model_type, args.model_path, args.num_classes, DEVICE)

    miou, mdice = evaluate(model, test_loader, DEVICE, args.num_classes)

    print(f"\nFinal Metrics for {args.model_type} ({args.model_path}):")
    print(f"  mIoU: {miou:.4f}")
    print(f"  mDice/mF1: {mdice:.4f}") 

    # add saving metrics to a file once this is known
