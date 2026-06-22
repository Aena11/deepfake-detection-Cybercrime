import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

IMG_SIZE, BATCH_SIZE, NUM_WORKERS = 224, 64, 4

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FaceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        real_imgs = [(p, 0) for p in Path(real_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        fake_imgs = [(p, 1) for p in Path(fake_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        self.samples = real_imgs + fake_imgs
        print(f'  Real: {len(real_imgs)} | Fake: {len(fake_imgs)} | Total: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

print('\nLoading model...')
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(device)
model.load_state_dict(torch.load('/speed-scratch/ae_verma/best_efficientnet_b4.pth', map_location=device))
model.eval()
print('Model loaded!')

datasets = {
    'FF++':          ('/speed-scratch/inse6610team2/ff_faces/real',              '/speed-scratch/inse6610team2/ff_faces/fake'),
    'CelebDF_V2':    ('/speed-scratch/inse6610team2/Celeb_V2/Test/real',         '/speed-scratch/inse6610team2/Celeb_V2/Test/fake'),
    'Sarra_Dataset': ('/speed-scratch/inse6610team2/dataset_images/real',        '/speed-scratch/inse6610team2/dataset_images/fake'),
    'CIFAKE':        ('/speed-scratch/inse6610team2/CIFAKE-ResNet-50/test/REAL', '/speed-scratch/inse6610team2/CIFAKE-ResNet-50/test/FAKE'),
    'Client_Dataset':('/speed-scratch/inse6610team2/real_vs_fake/real-vs-fake/test/real', '/speed-scratch/inse6610team2/real_vs_fake/real-vs-fake/test/fake'),
}

results = {}

for name, (real_dir, fake_dir) in datasets.items():
    print(f'\n{"="*50}')
    print(f'Testing on: {name}')
    ds = FaceDataset(real_dir, fake_dir, transform)
    if len(ds) == 0:
        print('  No images found, skipping!')
        continue
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs)).cpu().squeeze(1)
            preds = (outputs >= 0.5).float()
            all_probs.extend(outputs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f'  Accuracy : {acc:.4f}')
    print(f'  AUC-ROC  : {auc:.4f}')
    results[name] = {'accuracy': acc, 'auc': auc}
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    plt.title(f'Confusion Matrix - EfficientNet-B4 on {name}')
    plt.tight_layout()
    plt.savefig(f'/speed-scratch/ae_verma/cm_{name}.png', dpi=150)
    plt.close()

print(f'\n{"="*50}')
print('FINAL SUMMARY - EfficientNet-B4 Cross-Dataset Results')
print(f'{"="*50}')
print(f'{"Dataset":<20} {"Accuracy":>10} {"AUC-ROC":>10}')
print(f'{"-"*40}')
for name, r in results.items():
    print(f'{name:<20} {r["accuracy"]:>10.4f} {r["auc"]:>10.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
aucs = [results[n]['auc'] for n in names]
ax1.bar(names, accs, color='steelblue')
ax1.set_title('Accuracy per Dataset')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=15)
ax2.bar(names, aucs, color='darkorange')
ax2.set_title('AUC-ROC per Dataset')
ax2.set_ylabel('AUC-ROC')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('/speed-scratch/ae_verma/cross_dataset_results.png', dpi=150)
print('\nAll results and plots saved!')
print('DONE!')
