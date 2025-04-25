import torch
from torch.utils.data import DataLoader

from src.weakly_supervised.resnet import MultiHeadResNet
from src.utils.dataset import TrainTestSplit
from src.MultiTargetOxfordPet import MultiTargetOxfordPet
from src.evaluate import evaluate


def evaluate_model(model_path, num_classes=3):
    """Evaluate a trained model on the test set using the repository's evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MultiHeadResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test dataset using the MultiTargetOxfordPet class from the repository
    full_dataset = MultiTargetOxfordPet()
    test_indices = TrainTestSplit(range(len(full_dataset)), 0.8)[1].indices
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Use the existing evaluation function from the repository
    miou, mdice = evaluate(model, test_loader, device, num_classes)
    
    return miou, mdice