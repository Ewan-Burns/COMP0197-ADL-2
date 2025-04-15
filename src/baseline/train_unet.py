import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from unet import UNet
from src.utils.dice_loss import dice_loss
from tqdm import tqdm
from src.MultiTargetOxfordPet import MultiTargetOxfordPet



def show_prediction(img_tensor, mask_pred, act_mask):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_pred.cpu().numpy()
    target = act_mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(target)
    plt.title("Target Segmentation")
    plt.axis("off")
    plt.show()


def TrainModel():

    # Preprocessing
    train_set = MultiTargetOxfordPet()
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device('cuda')
    model = UNet(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for images, masks in bar:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss += dice_loss(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(masks, 3).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "./models/unet_3_classes.pth")


def TestModel(model, train_set):
    device = torch.device("cuda")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for x in range(5):
            sample_img, act_mask = train_set[x]
            img = sample_img.unsqueeze(0).to(device)
            output = model(img)
            pred_mask = output.argmax(1).squeeze(0)
            show_prediction(sample_img, pred_mask, act_mask)

if __name__ == "__main__":

    TrainModel()

    model = UNet(n_classes=3)
    model.load_state_dict(torch.load("./models/unet_3_classes.pth"))

    train_set = MultiTargetOxfordPet()
    TestModel(model, train_set)
