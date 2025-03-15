import pyautogui
from pynput.mouse import Listener
from PIL import Image
import time

# 用于保存截图的编号
screenshot_counter = 1

# 用于获取截图范围
def select_screenshot_area():
    print("请按下鼠标左键来选择截图区域的左上角，右下角来完成选择")
    time.sleep(2)  # 给用户一点时间来准备

    coords = []

    def on_click(x, y, button, pressed):
        if pressed:
            coords.append((x, y))
            print(f"Mouse pressed at ({x}, {y})")
            if len(coords) == 2:
                return False  # 停止监听

    with Listener(on_click=on_click) as listener:
        listener.join()

    if len(coords) == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        print(f"左上角选择的位置: ({x1}, {y1})")
        print(f"右下角选择的位置: ({x2}, {y2})")
        return (x1, y1, x2, y2)
    else:
        print("未能正确选择截图区域")
        return None

# 获取鼠标点击事件
Pressed_x = 0
Pressed_y = 0
def on_click(x, y, button, pressed):
    global screenshot_counter
    global Pressed_x
    global Pressed_y

    if pressed:
        Pressed_x = x
        Pressed_y = y
        print(f"Mouse pressed at ({x}, {y})")
    else:
        print(f"Mouse released at ({x}, {y})")
        # 只有点击在截图范围内才进行截图
        if x1 <= x <= x2 and y1 <= y <= y2:
            # 记录鼠标点击位置
            print(f"Mouse clicked at ({x}, {y})")

            # 获取当前时间，用于截图文件名
            now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

            # 获取指定区域的截图
            screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

            # 保存截图，文件名格式：日期_时间_编号_左上角位置_右下角位置_点击位置_抬起位置.png
            screenshot.save(f"{now}_{screenshot_counter}_{x1}_{y1}_{x2}_{y2}_{Pressed_x}_{Pressed_y}_{x}_{y}.png")
            print(f"Screenshot saved as screenshot_{screenshot_counter}_{x}_{y}.png")

            # 增加截图编号
            screenshot_counter += 1
        else:
            print(f"Click at ({x}, {y}) is outside the screenshot area.")

# 获取截图区域
screenshot_area = select_screenshot_area()
if screenshot_area:
    x1, y1, x2, y2 = screenshot_area

    # 启动监听鼠标点击事件
    with Listener(on_click=on_click) as listener:
        listener.join()
else:
    print("无法获取截图区域，程序退出")
