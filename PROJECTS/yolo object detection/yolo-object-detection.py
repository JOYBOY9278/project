#Import the necessary libraries
import numpy as np
import argparse
import cv2
import os

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run YOLO object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--config", default="yolov4.cfg",
                               help='Path to YOLO configuration file')
parser.add_argument("--weights", default="yolov4.weights",
                               help='Path to YOLO pre-trained weights')
parser.add_argument("--classes", default="coco.names",
                               help='Path to classes file')
parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Check if required files exist
required_files = {
    "Config": args.config,
    "Weights": args.weights,
    "Classes": args.classes
}

for file_type, file_path in required_files.items():
    if not os.path.exists(file_path):
        print(f"Error: {file_type} file '{file_path}' not found.")
        print("\nPlease download the required files:")
        print("1. YOLOv4 cfg: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        print("2. YOLOv4 weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        print("3. COCO names: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        exit(1)

# Load class names
classes = []
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# You can add your custom classes here
CUSTOM_CLASSES = {
    'my_custom_class1': {
        'id': 81,  # Start from 81 since COCO dataset uses 0-80
        'color': (0, 255, 0)  # BGR color for bounding box (green)
    },
    # Add more custom classes as needed
}

# Combine COCO classes with custom classes
classNames = { 
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    12: 'stop sign',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    28: 'tie',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    57: 'chair',
    60: 'bed',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    74: 'book',
    75: 'clock',
    78: 'teddy bear',
}

# Add custom classes to classNames
for class_name, class_info in CUSTOM_CLASSES.items():
    classNames[class_info['id']] = class_name

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get dimensions
    height, width, channels = frame.shape
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input to the network
    net.setInput(blob)
    
    # Run forward pass
    layer_outputs = net.forward(get_output_layers(net))
    
    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    
    # Process each detection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > args.thr and class_id in classNames:
                # Scale bounding box coordinates back to original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Add detection data to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args.thr, 0.4)
    
    # Draw detection results on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            
            # Draw bounding box
            # Use a default color or define a color mapping for each class
            color = (0, 255, 0)  # Default color (green)
            if class_ids[i] in CUSTOM_CLASSES:
                color = CUSTOM_CLASSES[classNames[class_ids[i]]]['color']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Put label and confidence
            if class_ids[i] in classNames:
                label = f"{classNames[class_ids[i]]}: {confidences[i]:.2f}"
                print(label)  # print class and confidence
                
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y = max(y, labelSize[1])
                cv2.rectangle(frame, (x, y - labelSize[1]), 
                                    (x + labelSize[0], y + baseLine),
                                    (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Display the resulting frame
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
