import torch
from torchvision import transforms

ResNetTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def TrainTestSplit(dataset, t, seed = 42):
    generator = torch.Generator().manual_seed(seed)
    train_size = int(t * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)