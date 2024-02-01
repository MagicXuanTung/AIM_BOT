from ultralytics import YOLO

#  just change this file to show data in weights  "yolov8x.pt"
model_weights_path = "../YOLO-Weights/best2024.pt"

# Load the YOLO model
all_objects_model = YOLO(model_weights_path)

# Display model architecture
print("Model Architecture:")
print(all_objects_model.model)

# Display input size
print("\nInput Size:")
print(all_objects_model.model.stride)

# Display class names
print("\nClass Names:")
print(all_objects_model.names)
