import cv2
import math
import pydirectinput
import pygetwindow as gw
import pyautogui
import numpy as np
from ultralytics import YOLO

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

# YOLO model initialization
model = YOLO("../YOLO-Weights/yolov8n.pt")

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

while True:
    # Capture the content of the specified window
    left, top, right, bottom = browser_window.left, browser_window.top, browser_window.right, browser_window.bottom
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    img = np.array(screenshot)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_without_boxes = img.copy()

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

                # Move the mouse cursor to the center of the bounding box with a duration of 0.5 seconds
                pydirectinput.moveTo((x1 + x2) // 2, (y1 + y2) // 2)  # Use pydirectinput instead of pyautogui

    cv2.putText(img, f'Person : {bounding_box_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cv2.destroyAllWindows()
