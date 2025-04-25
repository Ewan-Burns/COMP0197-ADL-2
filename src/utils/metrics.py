import torch
import torch.nn.functional as F
import numpy as np

def calculate_iou(pred, target, num_classes):
    """
    Calculates IoU for each class.
    Args:
        pred (torch.Tensor): Predicted segmentation map (N, H, W) or (H, W), integer values.
        target (torch.Tensor): Ground truth segmentation map (N, H, W) or (H, W), integer values.
        num_classes (int): Number of classes.
    Returns:
        np.ndarray: IoU for each class (shape [num_classes]).
    """
    iou_list = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflow
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            # if no ground truth or pred = perfect match
            # if there is ground truth but no prediction or vice versa, IoU is 0
            # handle case where union is 0
             iou = float('nan') 
        else:
            iou = float(intersection) / union
        iou_list.append(iou)

    # return numpy array, ignoring NaN values for mean calculation later
    return np.array(iou_list)


def calculate_miou(ious):
    """
    Calculates mean IoU, ignoring NaN values.
    Args:
        ious (list or np.ndarray): List or array of IoU values per class for multiple images.
                                   Each element can be a list/array of IoUs for one image.
    Returns:
        float: Mean IoU.
    """
    # flatten the 2d list and filter out NaNs
    all_ious = np.concatenate([iou for iou in ious if iou is not None], axis=0)
    valid_ious = all_ious[~np.isnan(all_ious)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
    return miou

def calculate_miou_per_image(pred, target, num_classes):
    """Calculates mIoU for a single image prediction."""
    iou_per_class = calculate_iou(pred, target, num_classes)
    miou = np.nanmean(iou_per_class) # nanmean ignores NaN values
    return miou if not np.isnan(miou) else 0.0


def calculate_dice(pred, target, num_classes, smooth=1e-6):
    """
    Calculates Dice score for each class.
    Args:
        pred (torch.Tensor): Predicted segmentation map (N, H, W) or (H, W), integer values.
        target (torch.Tensor): Ground truth segmentation map (N, H, W) or (H, W), integer values.
        num_classes (int): Number of classes.
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        np.ndarray: Dice score for each class (shape [num_classes]).
    """
    dice_list = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().item()
        sum_preds = pred_inds.long().sum().item()
        sum_targets = target_inds.long().sum().item()

        # Dice = 2 * |A intersect B| / (|A| + |B|)
        denominator = sum_preds + sum_targets
        if denominator == 0:
            # class not present in either pred or target, perfect score for this class
            dice = float('nan') # Or 1.0 depending on definition
        else:
            dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_list.append(dice)

    return np.array(dice_list)


def calculate_mdice(dices):
    """
    Calculates mean Dice score, ignoring NaN values.
    Args:
        dices (list or np.ndarray): List or array of Dice values per class for multiple images.
                                    Each element can be a list/array of Dice scores for one image.
    Returns:
        float: Mean Dice score.
    """
    # flatten the list of arrays/lists and filter out NaNs
    all_dices = np.concatenate([d for d in dices if d is not None], axis=0)
    valid_dices = all_dices[~np.isnan(all_dices)]
    mdice = np.mean(valid_dices) if len(valid_dices) > 0 else 0.0
    return mdice