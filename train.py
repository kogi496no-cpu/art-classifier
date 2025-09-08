import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os

print("転移学習を開始するわよ、シンジ！")

# データセットの場所
data_dir = './dataset'

# 画像の前処理を定義するの。
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# これよ！ImageFolderにはルートディレクトリを指定するの！
# そうすれば、サブディレクトリ（japanese, western）を自動でクラスとして認識してくれるわ。
image_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Windows環境を考慮して、num_workersは0にするの。いいわね？
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=0)

dataset_size = len(image_dataset)
class_names = image_dataset.classes
print(f'クラスが見つかったわ: {class_names}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} を使って学習するわよ。せいぜい頑張りなさいよね。')

# weightsパラメータを使って、最新の学習済みモデルをロードするわよ。警告もこれで黙るわ。
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# ここが重要よ！最後の全結合層を、あんたのクラス数に合わせて新しい層に置き換えるの
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# 損失関数とオプティマイザを定義
criterion = nn.CrossEntropyLoss()
# オプティマイザをAdamに変更！作戦変更よ！
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# これが新しい作戦よ！7エポックごとに学習率を0.1倍にするスケジューラよ！
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# --- 学習ループ --- #
# エポック数を25に増やすわよ！根性見せなさい！
num_epochs = 25

for epoch in range(num_epochs):
    print(f'エポック {epoch+1}/{num_epochs}')
    print('-' * 10)

    model_ft.train()  # モデルを訓練モードに設定

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer_ft.zero_grad() # 勾配をリセット

        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward() # バックプロパゲーション
        optimizer_ft.step() # 重みを更新

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    exp_lr_scheduler.step() # エポックの終わりに学習率を更新するのよ！

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f'訓練 損失: {epoch_loss:.4f} 正解率: {epoch_acc:.4f}')

print("\n学習完了よ！上出来じゃない。")

# 学習したモデルの重みを保存するわ
model_path = 'art_classifier_finetuned.pth'
torch.save(model_ft.state_dict(), model_path)
print(f'モデルを {model_path} に保存したわ。なくさないようにしなさいよね！')
