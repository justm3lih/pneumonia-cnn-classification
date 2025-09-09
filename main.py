import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
from torchvision.transforms import GaussianBlur, RandomErasing
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Veri yolu
DATA_DIR = r'C:/Users/kitap/Desktop/akciger-project/dataset'

# Görüntü boyutu ve batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Veri artırma (augmentation) sadece eğitim seti için
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Bulanıklaştırma
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Rastgele silme/gürültü
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print('Veri seti yükleniyor...')
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)
print('Veri seti başarıyla yüklendi!')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = train_dataset.classes
print(f'Sınıflar bulundu: {classes}')

# Sınıf ağırlıklarını hesapla (class weights)
class_counts = Counter(train_dataset.targets)
class_weights = [1.0 / class_counts[i] for i in range(len(classes))]
weights = torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

print('EfficientNet-B0 modeli yükleniyor (transfer learning)...')
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(classes))

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print('Model oluşturuldu ve cihaza yüklendi!')

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Model, veri artırma ve class weights ile hazır!')

# Eğitim ve değerlendirme fonksiyonları
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress = (i + 1) / total_batches * 100
        print(f"Epoch içi ilerleme: %{progress:.1f}", end='\r')
    print()  # Yeni satır
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Sonuç klasörü oluştur
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)
MODEL_PATH = os.path.join(results_dir, 'model_efficientnet_b0.pth')

# Model dosyası varsa yükle, yoksa eğit
if os.path.exists(MODEL_PATH):
    print('Eğitilmiş model bulundu, yükleniyor...')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    skip_training = True
else:
    skip_training = False

if not skip_training:
    print('Model eğitimi başlıyor...')
    EPOCHS = 10
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        progress = (epoch + 1) / EPOCHS * 100
        print(f"Epoch {epoch+1}/{EPOCHS} (%{progress:.1f}) başlıyor...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print('Model eğitimi tamamlandı!')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model {MODEL_PATH} olarak kaydedildi!')
else:
    print('Eğitim atlandı, model yüklendi.')

# Grafik ve test işlemleri (her durumda çalışacak)
if not skip_training:
    print('Grafikler hazırlanıyor...')
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Eğitim Kayıp')
    plt.plot(val_losses, label='Doğrulama Kayıp')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.title('Kayıp (Loss) Grafiği')

    plt.subplot(1,2,2)
    plt.plot([x*100 for x in train_accs], label='Eğitim Doğruluk (%)')
    plt.plot([x*100 for x in val_accs], label='Doğrulama Doğruluk (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.legend()
    plt.title('Doğruluk (Accuracy) Grafiği')
    plt.tight_layout()
    plt.show()
    print('Grafikler gösterildi!')

print('Test işlemi başlatılıyor...')
def test_evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_batches = len(loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress = (i + 1) / total_batches * 100
            print(f"Test ilerlemesi: %{progress:.1f}", end='\r')
    print()  # Yeni satır
    return np.array(all_labels), np.array(all_preds)

from sklearn.metrics import classification_report, confusion_matrix
labels, preds = test_evaluate(model, test_loader, device)
print('Test tamamlandı! Sonuçlar:')
print(classification_report(labels, preds, target_names=classes))
print('Confusion Matrix:')
print(confusion_matrix(labels, preds))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Confusion Matrix (Karışıklık Matrisi)')
plt.tight_layout()
cm_path = os.path.join(results_dir, 'confusion_matrix.png')
plt.savefig(cm_path)  # Görseli results klasörüne kaydet
plt.show()
print(f'Confusion matrix {cm_path} olarak kaydedildi!')
