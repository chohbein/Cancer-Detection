import pandas as pd
import numpy as np

import zipfile
import os

import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Subset, random_split

#           train folder
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

import json

from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

class ImageDF(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id']
        label = row['label']
        img_path = os.path.join(self.root_dir, img_id + ".tif")

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            image = torch.zeros((3, 46, 46))  # black dummy image

        return image, label

#   For predicting test set
class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        img_id = os.path.splitext(img_name)[0]

        return image, img_id



class TestImageDF(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.root_dir, img_id + ".tif")
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            image = torch.zeros((3, 46, 46))
        return image, img_id


import torch.nn.functional as F

#   Class for 1st Iteration
class CNNClassifier1(nn.Module):
    def __init__(self):
        super(CNNClassifier1, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128,kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256,kernel_size=3,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512,kernel_size=3,stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(0.3)  #   Dropout reduces overfitting
        self.fc1 = nn.Linear(512 * 2 * 2, 256)  #   Flatten 
        self.fc2 = nn.Linear(256, 1)    #   Flatten
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

#   Class for 2nd iteration
class CNNClassifier2(nn.Module):
    def __init__(self):
        super(CNNClassifier2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

#       Didnt work.
'''
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
'''

def transform():
    transform = transforms.Compose([
        transforms.Resize((46, 46)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load labels CSV
    labels_df = pd.read_csv('train/_labels.csv')

    # Create dataset instance
    df = ImageDF(dataframe=labels_df, root_dir='train', transform=transform)

    return df

def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()

            # Calculate loss
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item() * images.size(0)  # sum up loss * batch size
            total_samples += images.size(0)

            all_labels.append(labels.view(-1).detach().cpu())
            all_probs.append(probs.view(-1).detach().cpu())

    avg_val_loss = val_loss / total_samples
    return avg_val_loss, all_labels, all_probs


#       First iteration
def get_loader(iter_type,train_df,val_df):
    if iter_type == 1:
        model = CNNClassifier1().to(device)
    else:
        model = CNNClassifier2().to(device)

    train_loader = DataLoader(
        train_df,
        batch_size=64,  # Increase batch size for better GPU utilization
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True  # Reduces worker respawn overhead
    )

    val_loader = DataLoader(
        val_df,
        batch_size = 64,
        shuffle = True,
        num_workers = 6,
        pin_memory = True,
        persistent_workers = True
    )

    return train_loader,val_loader,model

if __name__ == "__main__":
    #   Enable GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

    #   Define which iteration to do
    #       1 -> detailed focus
    #       2 -> broader trends

    for t in [1,2]:
        iter_type = t

        df = transform()

        #   Train/Val Split
        df_size = len(df)
        train_size = int(0.8 * df_size)
        val_size = df_size - train_size

        train_df, val_df = random_split(df, [train_size, val_size], generator=torch.Generator().manual_seed(17))
        
        train_loader,val_loader,model = get_loader(iter_type,train_df,val_df)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)   #   Adam

        #   LR Scheduler, avoids overfitting
        #scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

        #   Training
        num_epochs = 10

        train_losses = []
        val_losses = []
        #early_stopping = EarlyStopping(patience=1, min_delta=0.001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)  # batch loss

            avg_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Evaluate on validation
            model.eval()

            avg_val_loss, all_labels, all_probs = evaluate_model(model, val_loader, device, criterion)
            val_losses.append(avg_val_loss)

            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            #early_stopping(avg_val_loss)

            roc_auc = roc_auc = roc_auc_score(torch.cat(all_labels).cpu().numpy(), torch.cat(all_probs).cpu().numpy())
            y_true = torch.cat(all_labels).cpu().numpy()
            y_prob = torch.cat(all_probs).cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

            print(roc_auc)
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(cm)

            #   Early Stopping (did not improve performance)
            '''
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            '''

            #   LR Scheduling (didn't improve performance)
            #scheduler.step()

        # Save to JSON
        losses = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        with open(f"losses_i{iter_type}.json", "w") as f:
            json.dump(losses, f)
    torch.save(model.state_dict(), "model_weights.pth")
    
    #   Predictions on Test Set (Kaggle)
    model = CNNClassifier2()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
    transforms.Resize((46, 46)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = TestImageDataset(root_dir='test', transform=test_transform)
    print(test_dataset[1])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

    model.eval()
    all_img_ids = []
    all_preds = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds = (np.array(probs) >= 0.5).astype(int)
            all_img_ids.extend(img_ids)
            all_preds.extend(preds)

    submission = pd.DataFrame({'id': all_img_ids, 'label': all_preds})
    submission.to_csv('submission.csv', index=False)
    #"""


