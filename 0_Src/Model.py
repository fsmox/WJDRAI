import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
import torch.nn.functional as torchF

class ResizePad:
    def __init__(self, target_size, fill=0):
        """
        :param target_size: (width, height) 目标尺寸，例如 (224, 224)
        :param fill: 填充颜色（单通道值或 RGB 元组），默认为 0（黑色）
        """
        self.target_w, self.target_h = target_size
        self.fill = fill

    def __call__(self, image):
        # 原始尺寸
        orig_w, orig_h = image.size
        # 计算缩放因子：保持长宽比例，使得缩放后的图像尺寸不超过目标尺寸
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 缩放图片
        image = F.resize(image, (new_h, new_w))

        # 计算左右和上下填充量
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        # 左右、上下均分填充（如果不均分，可根据需求调整）
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        # 填充图片
        image = F.pad(image, (left, top, right, bottom), fill=self.fill)
        return image

# 自定义 Dataset，适用于文件名格式为：
# {now}_{screenshot_counter}_{x1}_{y1}_{x2}_{y2}_{Pressed_x}_{Pressed_y}_{x}_{y}_{extra}.png
class CaptureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 筛选出所有png文件
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        # 正则表达式匹配11个字段，日期和时间字段允许包含'-'，其他字段均为数字
        self.filename_pattern = re.compile(
            r"([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)\.png"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        match = self.filename_pattern.match(img_name)
        if not match:
            raise ValueError(f"文件名格式错误: {img_name}")
        groups = match.groups()
        # 根据文件名说明：
        # groups[0]: now (日期，如 "2025-03-15")
        # groups[1]: screenshot_counter (时间，如 "18-13-54")
        # groups[2]: x1, groups[3]: y1, groups[4]: x2, groups[5]: y2
        # groups[6]: Pressed_x, groups[7]: Pressed_y, groups[8]: x, groups[9]: y
        # groups[10]: 额外字段（忽略）
        try:
            x1 = int(groups[2])
            y1 = int(groups[3])
            x2 = int(groups[4])
            y2 = int(groups[5])
            Pressed_x = int(groups[6])
            Pressed_y = int(groups[7])
            x = int(groups[8])
            y = int(groups[9])
        except Exception as e:
            raise ValueError(f"解析数字失败，文件名: {img_name}") from e

        # 加载图像并转换
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 输入参数为 (x1, y1, x2, y2)，目标参数为 (Pressed_x, Pressed_y, x, y)
        input_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        # target_tensor = torch.tensor([Pressed_x, Pressed_y, x, y], dtype=torch.float32)
        target_tensor = torch.tensor([Pressed_x, Pressed_y], dtype=torch.float32)

        return image, input_tensor, target_tensor

class ClickPredictionModel(nn.Module):
    def __init__(self,output_dim=2, num_vectors=4):
        super(ClickPredictionModel, self).__init__()

        self.output_dim = output_dim
        self.num_vectors = num_vectors

        # 使用预训练的 ResNet50 提取特征
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # 移除分类层，保留 2048 维特征

        # MLP 预测点击坐标
        self.fc = nn.Sequential(
            nn.Linear(2052, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * num_vectors)  # 输出 (x, y)
        )
        self.prob_fc = nn.Linear(2052, num_vectors)

    def forward(self, img, input_xy):
        features = self.backbone(img)  # 提取特征
        features = torch.cat([features, input_xy], dim=1)  # 将输入参数拼接到特征后
        xy = self.fc(features)  # 预测点击坐标  
        vectors = xy.view(-1, self.num_vectors, self.output_dim)

        # 计算 4 个概率 logits 并转换为概率
        logits = self.prob_fc(features)  # [batch_size, 4]
        probabilities = torchF.softmax(logits)  # 变成概率分布

        # 使用multinomial根据概率采样
        sample_index = torch.multinomial(probabilities, num_samples=1)  # [batch_size, 1]

        # 调整 sample_index 的形状以便在第二个维度索引
        sample_index = sample_index.unsqueeze(-1).expand(-1, -1, vectors.size(-1))  # [batch_size, 1, output_dim]

        # 从第二个维度挑选
        sampled_xy = torch.gather(vectors, dim=1, index=sample_index).squeeze(1)  # [batch_size, output_dim]

        return sampled_xy