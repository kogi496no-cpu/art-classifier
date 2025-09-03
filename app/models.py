import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# クラス名を定義しておくのよ。訓練時に確認した通り、アルファベット順よ
class_names = ['japanese', 'western']

# まずはモデルの骨格を定義するの。ResNet18よ
model = models.resnet18(pretrained=False) # もう学習済みモデルは使わないわ

# 最後の層を、あんたの2クラス分類用に付け替える
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# そして、アタシたちが育てたモデルの重みを読み込むのよ！
model_path = 'art_classifier_finetuned.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval() # 評価モードにしなさい

# 画像の前処理は訓練時と同じよ
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# 予測結果を返す関数。今度はちゃんとクラス名を返すわよ
def get_prediction(image_path):
    tensor = transform_image(image_path)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_index = torch.max(outputs, 1)
    # 予測されたインデックスを、クラス名に変換して返すの
    return class_names[predicted_index.item()]