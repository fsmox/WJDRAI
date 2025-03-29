from Model import CaptureDataset, ResizePad,ClickPredictionModel
from screenshot_too import select_screenshot_area
import time
import random
from PIL import ImageGrab
import pyautogui
import torch
from torchvision import transforms

import keyboard  # 需要先安装：pip install keyboard

screenshot_area = select_screenshot_area()
x1, y1, x2, y2 = screenshot_area
print(f"Selected screenshot area: ({x1}, {y1}) to ({x2}, {y2})")


use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
    print("Using CUDA")
elif use_mps:
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the pre-trained ClickPredictionModel
model_path = "Src/Model/click_prediction_model.pth"
model = ClickPredictionModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

input_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).unsqueeze(0)
ByInput = False
# 通过键盘输入坐标
while True:

    if ByInput:
        predicted_x, predicted_y = map(int, input("请输入坐标（x y）：").split())
        pressed_x, pressed_y = predicted_x + x1, predicted_y + y1
        # Check if the coordinates are within the selected range
        if not (x1 <= pressed_x <= x2 and y1 <= pressed_y <= y2):
            print("The predicted coordinates are outside the selected range. Ignoring.")
            continue
        pyautogui.moveTo(pressed_x, pressed_y)
        pyautogui.click()
        print(f"Clicked at: ({pressed_x}, {pressed_y})")

        continue

    # Capture the screenshot
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    # Convert the screenshot to a format suitable for the model
    screenshot = screenshot.convert("RGB")  # Ensure it's in RGB format
    
    # 定义图像预处理（调整尺寸和归一化，符合 ResNet50 要求）
    transform = transforms.Compose([
        ResizePad((224, 224), fill=(0, 0, 0)),  # 使用黑色填充
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    # screenshot = ResizePad((224, 224))(screenshot)  # Resize and pad to model's input size
    screenshot_tensor = transform(screenshot)  # Transform to tensor

    # Add batch dimension and pass through the model
    screenshot_tensor = screenshot_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(screenshot_tensor, input_tensor)  # Pass x1, y1 to the model
    # Extract the predicted click coordinates
    predicted_x, predicted_y = prediction[0][0].item(), prediction[0][1].item()

    print(f"Predicted coordinates: ({predicted_x}, {predicted_y})")
    
    # Convert the predicted coordinates to the original screen coordinates
    predicted_x = int(predicted_x) + x1
    predicted_y = int(predicted_y) + y1


    # Ensure the predicted coordinates are within the selected range
    if not (x1 <= predicted_x <= x2 and y1 <= predicted_y <= y2):
        print(f"Prediction: {prediction = }")
        print("The predicted coordinates are outside the selected range. Ignoring.")
    else:
        # Move the mouse to the predicted position and click
        pyautogui.moveTo(predicted_y, predicted_x)
        pyautogui.click()
        print(f"Prediction: {prediction = }")

    # Wait for 1 second plus a random delay
    time.sleep(5 + random.uniform(0.1, 0.5))


    # Check for a stop condition (e.g., pressing 'q' to quit)
    if keyboard.is_pressed('q'):  # 检查 'q' 键是否被按下
        print("Stopping the script.")
        break