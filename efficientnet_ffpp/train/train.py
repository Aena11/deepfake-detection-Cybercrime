import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

DATA_DIR = Path('/speed-scratch/ae_verma/data/ff_faces')
IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LR = 224, 64, 10, 1e-4
NUM_WORKERS = 4

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=42):
        self.transform = transform
        real_imgs = [(p, 0) for p in list(Path(root_dir, 'real').glob('*.jpg'))[:3000]]
        fake_imgs = [(p, 1) for p in list(Path(root_dir, 'fake').glob('*.jpg'))[:3000]]
        all_imgs = real_imgs + fake_imgs
        np.random.seed(seed)
        idx = np.random.permutation(len(all_imgs))
        n = len(idx)
        n_test = int(n * 0.15)
        n_val = int(n * 0.15)
        if split == 'test':
            chosen = idx[:n_test]
        elif split == 'val':
            chosen = idx[n_test:n_test + n_val]
        else:
            chosen = idx[n_test + n_val:]
        self.samples = [all_imgs[i] for i in chosen]
        print(f'{split}: {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = FaceDataset(DATA_DIR, 'train', train_transform)
val_ds = FaceDataset(DATA_DIR, 'val', val_transform)
test_ds = FaceDataset(DATA_DIR, 'test', val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1).to(device)
for name, param in model.named_parameters():
    if 'blocks.0' in name or 'blocks.1' in name or 'blocks.2' in name:
        param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-6)
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Train]'):
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Val]'):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    val_loss = running_loss / total
    val_acc = correct / total
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/speed-scratch/ae_verma/best_efficientnet_b4.pth')
        print(f'  Best model saved (val_acc={val_acc:.4f})')

print(f'Best Val Accuracy: {best_val_acc:.4f}')
model.load_state_dict(torch.load('/speed-scratch/ae_verma/best_efficientnet_b4.pth'))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc='Testing'):
        imgs = imgs.to(device)
        outputs = torch.sigmoid(model(imgs)).cpu().squeeze(1)
        preds = (outputs >= 0.5).float()
        all_probs.extend(outputs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
print(f'Test Accuracy : {acc:.4f}')
print(f'Test AUC      : {auc:.4f}')
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, NUM_EPOCHS+1), history['train_loss'], label='Train')
ax1.plot(range(1, NUM_EPOCHS+1), history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(range(1, NUM_EPOCHS+1), history['train_acc'], label='Train')
ax2.plot(range(1, NUM_EPOCHS+1), history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig('/speed-scratch/ae_verma/training_curves.png', dpi=150)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
plt.title('Confusion Matrix - EfficientNet-B4 on FF++')
plt.tight_layout()
plt.savefig('/speed-scratch/ae_verma/confusion_matrix.png', dpi=150)
print('DONE! Results saved.')
