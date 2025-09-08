
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

print("t-SNEによる特徴の可視化を開始するわよ！")

# --- パラメータ設定 ---
# 学習済みモデルのパス
MODEL_PATH = 'art_classifier_finetuned.pth'
# データセットの場所
DATA_DIR = './dataset'
# 出力する画像ファイル名
OUTPUT_IMAGE = 'tsne_visualization.png'

# --- モデルとデータセットの準備 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'使うデバイス: {device}')

# 画像の前処理：学習時と同じものを使うのが鉄則よ
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ImageFolderでデータセットを読み込み
image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=0)
class_names = image_dataset.classes
print(f'クラスを検出したわ: {class_names}')

# モデルの準備
if not os.path.exists(MODEL_PATH):
    print(f"エラー: モデルファイル '{MODEL_PATH}' が見つからないわ！")
    print("先に `train.py` を実行して、モデルを学習させなさい！")
    exit()

# 学習済みのResNet18モデルをロード
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# ここがキモよ！最終層を引っこ抜いて、特徴抽出器にするの
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval() # 評価モードに設定

# --- 特徴抽出 ---
print("全画像から特徴を抽出中... ちょっと待ってなさいよね。")
features_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        
        # 特徴を抽出
        features = feature_extractor(inputs)
        # (batch_size, num_features, 1, 1) -> (batch_size, num_features) に変形
        features = features.view(features.size(0), -1)
        
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# リストをNumPy配列に変換
features_array = np.concatenate(features_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)
print(f"特徴抽出完了！ {len(features_array)} 個の画像から特徴を抜き出したわ。")

# --- t-SNEの実行 ---
print("t-SNEで次元削減中... これが結構時間がかかるのよね。")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_array)-1))
features_2d = tsne.fit_transform(features_array)
print("t-SNE完了！")

# --- プロット ---
print("結果をプロットして、画像として保存するわよ。")
plt.figure(figsize=(12, 10))
palette = sns.color_palette("hsv", len(class_names))

sns.scatterplot(
    x=features_2d[:, 0], 
    y=features_2d[:, 1],
    hue=[class_names[label] for label in labels_array],
    palette=palette,
    legend="full",
    alpha=0.7
)

plt.title('t-SNEによる美術様式の可視化')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.legend(title='Art Style')
plt.savefig(OUTPUT_IMAGE)

print(f"完璧ね！可視化結果を '{OUTPUT_IMAGE}' に保存したわ。確認しなさい！")
