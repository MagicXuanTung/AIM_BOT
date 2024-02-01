import cv2
import time  # Import the time module
import math
import pydirectinput
import pygetwindow as gw
import pyautogui
import numpy as np
import torch
from ultralytics import YOLO

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.set_device(0)  # You may need to adjust the GPU device index (0, 1, etc.)

def find_window_by_title(title):
    try:
        return gw.getWindowsWithTitle(title)[0]
    except IndexError:
        return None

# Specify the title of the browser or application window you want to detect
window_title = "Edge"

# Find the window handle of the specified application
browser_window = find_window_by_title(window_title)
if browser_window:
    browser_window.minimize()
    browser_window.restore()
    browser_window.activate()
else:
    print(f"{window_title} window not found.")
    exit()

# Set your desired width and height
frame_width = 800
frame_height = 600

# Create an OpenCV window with the specified name
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", frame_width, frame_height)

# YOLO model initialization with GPU support
model = YOLO("../YOLO-Weights/yolov8n.pt").to(device)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Flags to control detection and mouse movement
detect_flag = True
mouse_move_flag = True
click_cooldown = 2.0  # Set the cooldown time in seconds
last_click_time = time.time() - click_cooldown  # Initialize last click time

while True:
    # Capture the content of the specified window
    left, top, right, bottom = browser_window.left, browser_window.top, browser_window.right, browser_window.bottom
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    img = np.array(screenshot)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_without_boxes = img.copy()

    if detect_flag:
        # Use YOLO model for object detection
        results = model(img, stream=True)

        bounding_box_count = 0

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                if classNames[cls] == "person":
                    bounding_box_count += 1

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    conf = math.ceil((box.conf[0] * 500)) / 500
                    class_name = classNames[cls]
                    label = f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                    # Move the mouse cursor to the center of the bounding box if mouse_move_flag is True
                    if mouse_move_flag:
                        pydirectinput.moveTo((x1 + x2) // 2, (y1 + y2) // 2)  # Use pydirectinput instead of pyautogui

                        # Check the cooldown period before clicking
                        current_time = time.time()
                        if current_time - last_click_time >= click_cooldown:
                            # Click with the left mouse button
                            pydirectinput.click()
                            last_click_time = current_time  # Update last click time

    cv2.putText(img, f'Person : {bounding_box_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Exit on Esc key
        break
    elif key == 112:  # Toggle detection on 'p' key
        detect_flag = not detect_flag
    elif key == 113:  # Toggle mouse movement on 'q' key
        mouse_move_flag = not mouse_move_flag

cv2.destroyAllWindows()


