import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from PIL import Image

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
        target_tensor = torch.tensor([Pressed_x, Pressed_y, x, y], dtype=torch.float32)

        return image, input_tensor, target_tensor

def main():
    # 获取当前脚本所在目录，并拼接 Capture 文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    capture_folder = os.path.join(current_dir, "Data")
    
    # 定义图像预处理（调整尺寸和归一化，符合 ResNet50 要求）
    transform = transforms.Compose([
        ResizePad((224, 224), fill=(0, 0, 0)),  # 使用黑色填充
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化 Dataset 和 DataLoader
    dataset = CaptureDataset(root_dir=capture_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 测试 DataLoader 是否正确加载数据
    for images, inputs, targets in dataloader:
        print("Image batch shape:", images.shape)   # 应为 (batch_size, 3, 224, 224)
        print("Input params shape:", inputs.shape)    # 应为 (batch_size, 4)
        print("Target shape:", targets.shape)         # 应为 (batch_size, 4)

        # 显示第一张图片
        img = images[0].permute(1, 2, 0).numpy()  # 调整维度顺序以适应 matplotlib
        img = (img * 255).astype('uint8')  # 反归一化并转换为 uint8 类型
        plt.imshow(img)
        plt.show()
        break

if __name__ == '__main__':
    main()

# class ClickPredictionModel(nn.Module):
#     def __init__(self):
#         super(ClickPredictionModel, self).__init__()
#         # 使用预训练的 ResNet50 提取特征
#         self.backbone = models.resnet50(pretrained=True)
#         self.backbone.fc = nn.Identity()  # 移除分类层，保留 2048 维特征

#         # MLP 预测点击坐标
#         self.fc = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2)  # 输出 (x, y)
#         )

#     def forward(self, img):
#         features = self.backbone(img)  # 提取特征
#         xy = self.fc(features)  # 预测点击坐标
#         return xy

# loss_fn = nn.MSELoss()
# loss = loss_fn(predicted_xy, ground_truth_xy)

# loss_fn = nn.SmoothL1Loss()
# loss = loss_fn(predicted_xy, ground_truth_xy)

# import torch.optim as optim

# # 创建模型
# model = ClickPredictionModel().cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# for epoch in range(epochs):
#     for img, target_xy in dataloader:  # img: (batch, 3, H, W), target_xy: (batch, 2)
#         img, target_xy = img.cuda(), target_xy.cuda()
#         optimizer.zero_grad()
#         output_xy = model(img)
#         loss = loss_fn(output_xy, target_xy)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch}, Loss: {loss.item()}")