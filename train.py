import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# DATASET
class TreeDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

        print(" Total images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        
        if image is None or mask is None:
            return self.__getitem__((idx + 1) % len(self.images))

        image = cv2.resize(image, (256, 256)) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.resize(mask, (256, 256)) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# MODEL
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )

        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)

        x4 = self.up(x3)
        x5 = torch.cat([x4, x1], dim=1)

        return self.dec(x5)   

# TRAIN
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TreeDataset("dataset/train/images", "dataset/train/masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()   # improved loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 20

    print(" Training started...")

    for epoch in range(epochs):
        total_loss = 0

        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        print(f" Epoch {epoch+1} Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "tree_model.pth")
    print(" Model saved successfully!")


if __name__ == "__main__":
    train_model()